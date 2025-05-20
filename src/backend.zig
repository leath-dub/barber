const std = @import("std");
const log = std.log;
const fs = std.fs;
const mem = std.mem;

const wayland = @import("wayland");
const wl = wayland.client.wl;

const vk = @import("vk.zig");
const mesh = @import("mesh.zig");
const Device = @import("device.zig");
const Vert = mesh.Vert;
const Verts = mesh.Verts;
const Mesh = mesh.Mesh;
const VertexBuffer = mesh.VertexBuffer;

const c = @import("c.zig").includes;

const zmesh = @import("zmesh");

pub const VulkanContext = @This();

arena: std.heap.ArenaAllocator,
temp_allocator: std.mem.Allocator,
surface: c.VkSurfaceKHR,
instance: c.VkInstance,
device: Device,

swapchain: Swapchain,

command_pool: c.VkCommandPool,
command_buffers: []c.VkCommandBuffer,
render_complete: c.VkSemaphore,
present_complete: c.VkSemaphore,

vertex_shader: c.VkShaderModule,
fragment_shader: c.VkShaderModule,

render_pass: c.VkRenderPass,
pipeline: c.VkPipeline,
pipeline_layout: c.VkPipelineLayout,

fences: []c.VkFence,
semaphores: []c.VkSemaphore,
frame: usize = 0,

object: Mesh,

messenger: c.VkDebugUtilsMessengerEXT,

const DEFAULT_FORMAT: c.VkFormat = c.VK_FORMAT_R8G8B8A8_UNORM;
const DEFAULT_COLORSPACE: c.VkColorSpaceKHR = c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

const VERTEX_SHADER_DATA = @embedFile("shaders/vertex.glsl");
const FRAGMENT_SHADER_DATA = @embedFile("shaders/fragment.glsl");

const MAX_FRAMES_IN_FLIGHT = 2;

fn debugCallback(serverity: c.VkDebugUtilsMessageSeverityFlagBitsEXT, message_type: c.VkDebugUtilsMessageTypeFlagsEXT, callback_data: [*c]const c.VkDebugUtilsMessengerCallbackDataEXT, user_data: ?*anyopaque) callconv(.c) c.VkBool32 {
    _ = message_type;
    _ = user_data;
    const severity_str = switch (serverity) {
        c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT => "error",
        c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT => "verbose",
        c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT => "info",
        c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT => "warning",
        else => unreachable,
    };
    log.debug("-- VULKAN ({s}) -- {s}", .{severity_str, callback_data.*.pMessage});
    return c.VK_FALSE;
}

pub fn init(safe_allocator: std.mem.Allocator, temp_allocator: std.mem.Allocator, width: u32, height: u32, surface: *wl.Surface, display: *wl.Display) !VulkanContext {
    var arena = std.heap.ArenaAllocator.init(safe_allocator);
    errdefer arena.deinit();

    const enabled = [_]c.VkValidationFeatureEnableEXT{c.VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};
    var features = vk.SType(c.VkValidationFeaturesEXT, .{
        .enabledValidationFeatureCount = 1,
        .pEnabledValidationFeatures = &enabled,
    });

    const instance_extensions = [_][*:0]const u8{ "VK_KHR_wayland_surface", "VK_KHR_surface", "VK_EXT_debug_utils" };
    var instance_create_info = vk.SType(c.VkInstanceCreateInfo, .{
        .pApplicationInfo = &vk.SType(c.VkApplicationInfo, .{
            .pApplicationName = "barber",
            .applicationVersion = comptime c.VK_MAKE_VERSION(0, 0, 1),
            .apiVersion = comptime c.VK_MAKE_VERSION(1, 3, 0),
        }),
        .enabledExtensionCount = instance_extensions[0..].len,
        .ppEnabledExtensionNames = &instance_extensions,
    });

    features.pNext = instance_create_info.pNext;
    instance_create_info.pNext = &features;

    var instance: c.VkInstance = undefined;
    try vk.check(c.vkCreateInstance(&withLayerIfAvailable(temp_allocator, "VK_LAYER_KHRONOS_validation", instance_create_info), null, &instance));
    log.info("vulkan version 1.3 instance acquired", .{});

    const debug_utils_info = vk.SType(c.VkDebugUtilsMessengerCreateInfoEXT, .{
        .messageSeverity = c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT,
        .messageType = c.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT | c.VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT,
        .pfnUserCallback = debugCallback,
    });
    var messenger: c.VkDebugUtilsMessengerEXT = undefined;

    const vkCreateDebugUtilsMessengerEXT: c.PFN_vkCreateDebugUtilsMessengerEXT = @ptrCast(c.vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    try vk.check(vkCreateDebugUtilsMessengerEXT.?(instance, &debug_utils_info, null, &messenger));
 

    const surface_create_info = vk.SType(c.VkWaylandSurfaceCreateInfoKHR, .{
        .display = @ptrCast(display),
        .surface = @ptrCast(surface),
    });
    var vk_surface: c.VkSurfaceKHR = undefined;
    try vk.check(c.vkCreateWaylandSurfaceKHR(instance, &surface_create_info, null, &vk_surface));
 
    const device = Device.select(temp_allocator, instance, vk_surface) orelse {
        log.err("failed to find suitable vulkan device. need version >=1.3 with graphics and bgra image format", .{});
        return error.NoSuitableVulkanDevice;
    };
    errdefer device.deinit();

    // Setup vertices
    
    // Create a render pass
    // Attachments are dependencies between rendering steps (e.g. textures)
    var attachment = mem.zeroInit(c.VkAttachmentDescription, .{
        .format = DEFAULT_FORMAT,
        .samples = c.VK_SAMPLE_COUNT_1_BIT,
        .loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = c.VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    });
    const attachment_ref = mem.zeroInit(c.VkAttachmentReference, .{ .attachment = 0, .layout = c.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
    var subpass = mem.zeroInit(c.VkSubpassDescription, .{
        .pipelineBindPoint = c.VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment_ref,
    });
    var render_pass_create_info = vk.SType(c.VkRenderPassCreateInfo, .{
        .attachmentCount = 1,
        .pAttachments = &attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
    });
    var render_pass: c.VkRenderPass = undefined;
    try vk.check(c.vkCreateRenderPass(device.logical, &render_pass_create_info, null, &render_pass));

    const swapchain = try Swapchain.init(safe_allocator, render_pass, width, height, device, vk_surface);

    var push: [6]f32 = undefined;
    const object = object: {
        var pool_info = vk.SType(c.VkCommandPoolCreateInfo, .{
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = device.getGraphicsQueueIndex()
        });
        var command_pool: c.VkCommandPool = undefined;
        try vk.check(c.vkCreateCommandPool(device.logical, &pool_info, null, &command_pool));
        defer c.vkDestroyCommandPool(device.logical, command_pool, null);

        var allocate_info = vk.SType(c.VkCommandBufferAllocateInfo, .{
            .commandPool = command_pool,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = @intCast(swapchain.images.len),
        });
        var copy_cmdbuf: c.VkCommandBuffer = undefined;
        try vk.check(c.vkAllocateCommandBuffers(device.logical, &allocate_info, &copy_cmdbuf));
        defer c.vkFreeCommandBuffers(device.logical, command_pool, 1, &copy_cmdbuf);

        var shape = zmesh.Shape.initCube();
        // Changes the coordanites to be in -1..1 range, not 0..1 range
        shape.scale(2, 2, 1);
        shape.translate(-1, -1, 0);
 
        push = shape.computeAabb();

        shape.computeNormals();
        defer shape.deinit();

        break :object try Mesh.init(device, copy_cmdbuf, .{
            .indices = shape.indices,
            .positions = shape.positions,
            .normals = shape.normals,
            .texcoords = shape.texcoords,
        });
    };

    // Setup a graphics pipeline
    const vertex_spirv = try vk.compileShader(temp_allocator, .vertex, VERTEX_SHADER_DATA);
    const fragment_spirv = try vk.compileShader(temp_allocator, .fragment, FRAGMENT_SHADER_DATA);

    var vertex_shader: c.VkShaderModule = undefined;
    var fragment_shader: c.VkShaderModule = undefined;
    try vk.check(c.vkCreateShaderModule(device.logical, &vk.SType(c.VkShaderModuleCreateInfo, .{
         // Even though it takes 32-bit int array it wants byte size - fucking stupid 
         // I think vulkan may have been designed by bill gates to waste Linux developers time
        .codeSize = vertex_spirv.len * 4,
        .pCode = vertex_spirv.ptr,
    }), null, &vertex_shader));
    try vk.check(c.vkCreateShaderModule(device.logical, &vk.SType(c.VkShaderModuleCreateInfo, .{
        .codeSize = fragment_spirv.len * 4,
        .pCode = fragment_spirv.ptr,
    }), null, &fragment_shader));

    const shader_stages = [_]c.VkPipelineShaderStageCreateInfo {
        vk.SType(c.VkPipelineShaderStageCreateInfo, .{ .stage = c.VK_SHADER_STAGE_VERTEX_BIT, .module = vertex_shader, .pName = "main" }),
        vk.SType(c.VkPipelineShaderStageCreateInfo, .{ .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragment_shader, .pName = "main" }),
    };

    var vertex_bindings: [VertexBuffer.Bindings.len]c.VkVertexInputBindingDescription = undefined;
    inline for (VertexBuffer.Bindings, 0..) |binding, i| {
        vertex_bindings[i] =
            .{ .binding = i, .stride = @intCast(VertexBuffer.sizeOf(binding)), .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX };
    }

    var vertex_attrs: [VertexBuffer.Bindings.len]c.VkVertexInputAttributeDescription = undefined;
    inline for (VertexBuffer.Bindings, 0..) |binding, i| {
        vertex_attrs[i] =
            .{ .binding = i, .location = i, .format = @field(VertexBuffer.binding_type, @tagName(binding)), .offset = 0 };
    }

    const vertex_input_info = vk.SType(c.VkPipelineVertexInputStateCreateInfo, .{
        .vertexBindingDescriptionCount = vertex_bindings.len,
        .pVertexBindingDescriptions = &vertex_bindings,
        .vertexAttributeDescriptionCount = vertex_attrs.len,
        .pVertexAttributeDescriptions = &vertex_attrs,
    });
    const input_assembly_info = vk.SType(c.VkPipelineInputAssemblyStateCreateInfo, .{ .topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST });
    const viewport_info = vk.SType(c.VkPipelineViewportStateCreateInfo, .{
        .viewportCount = 1,
        .pViewports = &c.VkViewport {
            .x = 0,
            .y = 0,
            .width = @floatFromInt(width),
            .height = @floatFromInt(height),
            .minDepth = 0,
            .maxDepth = 1,
        },
        .scissorCount = 1,
        .pScissors = &c.VkRect2D {
            .offset = .{ .x = 0, .y = 0 },
            .extent = .{ .height = height, .width = width },
        },
    });
    const raster_info = vk.SType(c.VkPipelineRasterizationStateCreateInfo, .{
        .polygonMode = c.VK_POLYGON_MODE_FILL,
        .cullMode = c.VK_CULL_MODE_NONE,
        .frontFace = c.VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1,
    });
    const multisample_info = vk.SType(c.VkPipelineMultisampleStateCreateInfo, .{
        .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT,
        .minSampleShading = 1,
    });
    const color_blend_info = vk.SType(c.VkPipelineColorBlendStateCreateInfo, .{
        .logicOp = c.VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &mem.zeroInit(c.VkPipelineColorBlendAttachmentState, .{
            .colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT,
        }),
    });

    const layout_info = vk.SType(c.VkPipelineLayoutCreateInfo, .{
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &.{ .offset = 0, .size = @sizeOf([6]f32), .stageFlags = c.VK_SHADER_STAGE_VERTEX_BIT },
    });
    var pipeline_layout: c.VkPipelineLayout = undefined;
    try vk.check(c.vkCreatePipelineLayout(device.logical, &layout_info, null, &pipeline_layout));

    const pipeline_info = vk.SType(c.VkGraphicsPipelineCreateInfo, .{
        .stageCount = @intCast(shader_stages.len),
        .pStages = &shader_stages,
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly_info,
        .pViewportState = &viewport_info,
        .pRasterizationState = &raster_info,
        .pMultisampleState = &multisample_info,
        .pColorBlendState = &color_blend_info,
        .layout = pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineIndex = -1,
    });

    var pipeline: c.VkPipeline = undefined;
    try vk.check(c.vkCreateGraphicsPipelines(device.logical, null, 1, &pipeline_info, null, &pipeline));

    // Setup command buffers

    var pool_info = vk.SType(c.VkCommandPoolCreateInfo, .{
        .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = device.getGraphicsQueueIndex()
    });
    var command_pool: c.VkCommandPool = undefined;
    try vk.check(c.vkCreateCommandPool(device.logical, &pool_info, null, &command_pool));

    var allocate_info = vk.SType(c.VkCommandBufferAllocateInfo, .{
        .commandPool = command_pool,
        .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = @intCast(swapchain.images.len),
    });
    var command_buffers = try arena.allocator().alloc(c.VkCommandBuffer, swapchain.images.len);
    try vk.check(c.vkAllocateCommandBuffers(device.logical, &allocate_info, command_buffers.ptr));
    command_buffers.len = swapchain.images.len;

    var render_pass_begin_info = vk.SType(c.VkRenderPassBeginInfo, .{
        .renderPass = render_pass,
        .renderArea = .{
            .offset = .{ .x = 0, .y = 0 },
            .extent = .{ .width = width, .height = height },
        },
        .clearValueCount = 1,
        .pClearValues = &c.VkClearValue { .color = .{ .float32 = .{0, 0, 0, 0} } },
    });

    for (command_buffers, 0..) |command_buffer, i| {
        var begin_info = vk.SType(c.VkCommandBufferBeginInfo, .{ .flags = c.VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT });
        try vk.check(c.vkBeginCommandBuffer(command_buffer, &begin_info));

        render_pass_begin_info.framebuffer = swapchain.framebuffers[i];
        c.vkCmdBeginRenderPass(command_buffer, &render_pass_begin_info, c.VK_SUBPASS_CONTENTS_INLINE);
        c.vkCmdBindPipeline(command_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        const vertex_data = object.buffers.values[VertexBuffer.Index.indices.ord() + 1..];
        c.vkCmdPushConstants(command_buffer, pipeline_layout, c.VK_SHADER_STAGE_VERTEX_BIT, 0, @sizeOf([6]f32), &push);
        c.vkCmdBindIndexBuffer(command_buffer, object.buffers.get(.indices), 0, c.VK_INDEX_TYPE_UINT32);
        c.vkCmdBindVertexBuffers(command_buffer, 0, VertexBuffer.Bindings.len, vertex_data.ptr, &std.mem.zeroes([VertexBuffer.Bindings.len]u64));
        c.vkCmdDrawIndexed(command_buffer, @intCast(object.sizes.get(.indices) / VertexBuffer.sizeOf(.indices)), 1, 0, 0, 0);
        c.vkCmdEndRenderPass(command_buffer);

        try vk.check(c.vkEndCommandBuffer(command_buffer));
    }

    var semaphore_info = vk.SType(c.VkSemaphoreCreateInfo, .{});
    const semaphores = try arena.allocator().alloc(c.VkSemaphore, MAX_FRAMES_IN_FLIGHT * 2);
    for (semaphores) |*semaphore| {
        try vk.check(c.vkCreateSemaphore(device.logical, &semaphore_info, null, semaphore));
    }
    const fences = try arena.allocator().alloc(c.VkFence, MAX_FRAMES_IN_FLIGHT);
    for (fences) |*fence| {
        try vk.check(c.vkCreateFence(device.logical, &vk.SType(c.VkFenceCreateInfo, .{ .flags = c.VK_FENCE_CREATE_SIGNALED_BIT }), null, fence));
    }

    var render_complete: c.VkSemaphore = undefined;
    var present_complete: c.VkSemaphore = undefined;
    try vk.check(c.vkCreateSemaphore(device.logical, &semaphore_info, null, &render_complete));
    try vk.check(c.vkCreateSemaphore(device.logical, &semaphore_info, null, &present_complete));

    return .{
        .arena = arena,
        .temp_allocator = temp_allocator,
        .surface = vk_surface,
        .instance = instance,
        .device = device,
        .swapchain = swapchain,
        .command_pool = command_pool,
        .command_buffers = command_buffers,
        .render_complete = render_complete,
        .present_complete = present_complete,
        .vertex_shader = vertex_shader,
        .fragment_shader = fragment_shader,
        .render_pass = render_pass,
        .pipeline = pipeline,
        .pipeline_layout = pipeline_layout,
        .fences = fences,
        .semaphores = semaphores,
        .frame = 0,
        .object = object,
        .messenger = messenger,
    };
}

pub fn tick(context: *VulkanContext) !void {
    const gq = context.device.getGraphicsQueue();
    const pq = context.device.getPresentQueue();

    const RENDER = 0;
    const PRESENT = 1;
    const semaphore_pair = context.semaphores[context.frame * 2..];

    try vk.check(c.vkWaitForFences(context.device.logical, 1, &context.fences[context.frame], c.VK_TRUE, std.math.maxInt(u64)));
    try vk.check(c.vkResetFences(context.device.logical, 1, &context.fences[context.frame]));

    var image_index: u32 = undefined;
    try vk.check(c.vkAcquireNextImageKHR(context.device.logical, context.swapchain.handle, std.math.maxInt(u32), semaphore_pair[PRESENT], null, &image_index));

    var submit_info = vk.SType(c.VkSubmitInfo, .{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &semaphore_pair[PRESENT],
        .pWaitDstStageMask = &@as(u32, c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
        .commandBufferCount = 1,
        .pCommandBuffers = &context.command_buffers[image_index],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &semaphore_pair[RENDER],
    });
    try vk.check(c.vkQueueSubmit(gq, 1, &submit_info, context.fences[context.frame]));

    var present_info = vk.SType(c.VkPresentInfoKHR, .{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &semaphore_pair[RENDER],
        .swapchainCount = 1,
        .pSwapchains = &context.swapchain.handle,
        .pImageIndices = &image_index,
    });
    try vk.check(c.vkQueuePresentKHR(pq, &present_info));

    context.frame = (context.frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

pub fn deinit(context: VulkanContext) void {
    context.object.deinit();

    c.vkDestroySurfaceKHR(context.instance, context.surface, null);
    c.vkDestroyShaderModule(context.device.logical, context.vertex_shader, null);
    c.vkDestroyShaderModule(context.device.logical, context.fragment_shader, null);
    for (context.semaphores) |semaphore| {
        c.vkDestroySemaphore(context.device.logical, semaphore, null);
    }
    for (context.fences) |fence| {
        c.vkDestroyFence(context.device.logical, fence, null);
    }
    c.vkFreeCommandBuffers(context.device.logical, context.command_pool, @intCast(context.command_buffers.len), context.command_buffers.ptr);
    c.vkDestroyCommandPool(context.device.logical, context.command_pool, null);
    context.swapchain.deinit(context);
    c.vkDestroyPipelineLayout(context.device.logical, context.pipeline_layout, null);
    c.vkDestroyPipeline(context.device.logical, context.pipeline, null);
    c.vkDestroyRenderPass(context.device.logical, context.render_pass, null);
    context.device.deinit();

    const vkDestroyDebugUtilsMessengerEXT: c.PFN_vkDestroyDebugUtilsMessengerEXT = @ptrCast(c.vkGetInstanceProcAddr(context.instance, "vkDestroyDebugUtilsMessengerEXT"));
    vkDestroyDebugUtilsMessengerEXT.?(context.instance, context.messenger, null);
    c.vkDestroyInstance(context.instance, null);
}

// This is separated from the context directly so that in the future we can separate the lifetime with its own arena.
// This struct stores things that are depended on by the swapchain so that when you re-create a swapchain you also re-create those.
const Swapchain = struct {
    arena: std.heap.ArenaAllocator, // for allocators that share the lifetime of the VulkanContext (usually static lifetime)
    handle: c.VkSwapchainKHR,
    images: []c.VkImage,
    views: []c.VkImageView,
    framebuffers: []c.VkFramebuffer, // I know that storing the framebuffers here may be a bit strange, but they do depend on the image views
    surface_caps: c.VkSurfaceCapabilitiesKHR,

    pub fn init(safe_allocator: std.mem.Allocator, render_pass: c.VkRenderPass, width: u32, height: u32, device: Device, surface: c.VkSurfaceKHR) !Swapchain {
        var arena = std.heap.ArenaAllocator.init(safe_allocator);
        errdefer arena.deinit();

        var surface_caps: c.VkSurfaceCapabilitiesKHR = undefined;
        try vk.check(c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.physical, surface, &surface_caps));
        if ((surface_caps.supportedCompositeAlpha & c.VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) == 0) return error.AlphaChannelNotSupported;

        var swapchain_create_info = vk.SType(c.VkSwapchainCreateInfoKHR, .{
            .surface = surface,
            .minImageCount = surface_caps.minImageCount,
            .imageFormat = DEFAULT_FORMAT,
            .imageColorSpace = DEFAULT_COLORSPACE,
            .imageExtent = .{ .width = width, .height = height },
            .imageArrayLayers = 1,
            .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .presentMode = c.VK_PRESENT_MODE_FIFO_KHR,
            .preTransform = surface_caps.currentTransform,
            .compositeAlpha = c.VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
            .clipped = c.VK_TRUE,
        });

        // Handle whether we have two different queues for graphics and presentation
        var q_indices: [2]u32 = undefined;
        switch (device.queue) {
            .single => {
                swapchain_create_info.imageSharingMode = c.VK_SHARING_MODE_EXCLUSIVE;
            },
            .split => |queue_pair| {
                swapchain_create_info.imageSharingMode = c.VK_SHARING_MODE_CONCURRENT;
                swapchain_create_info.queueFamilyIndexCount = 2;
                q_indices[0] = queue_pair.graphics.index;
                q_indices[1] = queue_pair.present.index;
                swapchain_create_info.pQueueFamilyIndices = &q_indices;
            },
        }

        var swapchain: c.VkSwapchainKHR = undefined;
        try vk.check(c.vkCreateSwapchainKHR(device.logical, &swapchain_create_info, null, &swapchain));

        var swapchain_images_count: u32 = 0;
        try vk.check(c.vkGetSwapchainImagesKHR(device.logical, swapchain, &swapchain_images_count, null));
        if (swapchain_images_count == 0) return error.NoSwapChainImages;

        var swapchain_images = try arena.allocator().alloc(c.VkImage, swapchain_images_count);
        try vk.check(c.vkGetSwapchainImagesKHR(device.logical, swapchain, &swapchain_images_count, swapchain_images.ptr));
        swapchain_images.len = swapchain_images_count;

        var image_view_create_info = vk.SType(c.VkImageViewCreateInfo, .{
            .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
            .format = swapchain_create_info.imageFormat,
            .subresourceRange = .{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1,
            },
        });
        var image_views = try arena.allocator().alloc(c.VkImageView, swapchain_images_count);
        for (swapchain_images, 0..) |image, i| {
            image_view_create_info.image = image;
            try vk.check(c.vkCreateImageView(device.logical, &image_view_create_info, null, &image_views[i]));
        }
        image_views.len = swapchain_images_count;

        var framebuffer_create_info = vk.SType(c.VkFramebufferCreateInfo, .{
            .renderPass = render_pass,
            .attachmentCount = 1,
            .width = width,
            .height = height,
            .layers = 1,
        });
        var framebuffers = try arena.allocator().alloc(c.VkFramebuffer, swapchain_images_count);
        for (0..swapchain_images_count) |i| {
            framebuffer_create_info.pAttachments = &image_views[i];
            try vk.check(c.vkCreateFramebuffer(device.logical, &framebuffer_create_info, null, &framebuffers[i]));
        }
        framebuffers.len = swapchain_images_count;

        return .{
            .arena = arena,
            .handle = swapchain,
            .images = swapchain_images,
            .views = image_views,
            .framebuffers = framebuffers,
            .surface_caps = surface_caps,
        };
    }

    pub fn deinit(swapchain: Swapchain, context: VulkanContext) void {
        for (swapchain.framebuffers) |framebuffer| {
            c.vkDestroyFramebuffer(context.device.logical, framebuffer, null);
        }
        for (swapchain.views) |view| {
            c.vkDestroyImageView(context.device.logical, view, null);
        }
        swapchain.arena.deinit();
        c.vkDestroySwapchainKHR(context.device.logical, swapchain.handle, null);
    }
};

fn hasLayer(arena: std.mem.Allocator, layer_name: []const u8) bool {
    var layer_count: u32 = undefined;
    vk.check(c.vkEnumerateInstanceLayerProperties(&layer_count, null)) catch unreachable;

    var layers = arena.alloc(c.VkLayerProperties, layer_count) catch unreachable;
    vk.check(c.vkEnumerateInstanceLayerProperties(&layer_count, layers.ptr)) catch unreachable;
    layers.len = layer_count;

    for (layers) |layer_props| {
        if (std.mem.eql(u8, layer_name, layer_props.layerName[0..layer_name.len])) {
            return true;
        }
    }

    return false;
}

fn withLayerIfAvailable(arena: std.mem.Allocator, layer_name: []const u8, create_info_: c.VkInstanceCreateInfo) c.VkInstanceCreateInfo {
    // This is scummy as fuck! I am storing a pointer sized variable statically so that we have
    // a stable reference assigned to `ppEnabledLayerNames`
    const Static = struct {
        var layerName: usize = undefined;
    };
    var create_info = create_info_;
    if (!hasLayer(arena, layer_name)) {
        log.info("validation layers not available, continuing without them", .{});
        return create_info;
    }
    create_info.enabledLayerCount = 1;
    Static.layerName = @intFromPtr(layer_name.ptr);
    create_info.ppEnabledLayerNames = @ptrCast(&Static.layerName);
    return create_info;
}
