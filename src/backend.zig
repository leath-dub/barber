const std = @import("std");
const log = std.log;
const fs = std.fs;

const wayland = @import("wayland");
const wl = wayland.client.wl;

const c = @cImport({
    @cInclude("vulkan/vulkan.h");
    @cInclude("vulkan/vulkan_wayland.h");
});

const vk = @import("vk.zig");

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

const VERTEX_SHADER_DATA = @embedFile("shaders/vertex.glsl");
const FRAGMENT_SHADER_DATA = @embedFile("shaders/fragment.glsl");

pub fn init(safe_allocator: std.mem.Allocator, temp_allocator: std.mem.Allocator, width: u32, height: u32, surface: *wl.Surface, display: *wl.Display) !VulkanContext {
    var arena = std.heap.ArenaAllocator.init(safe_allocator);
    errdefer arena.deinit();

    const instance_extensions = [_][*:0]const u8{ "VK_KHR_wayland_surface", "VK_KHR_surface" };
    const instance_create_info = vk.SType(c.VkInstanceCreateInfo, .{
        .pApplicationInfo = &vk.SType(c.VkApplicationInfo, .{
            .pApplicationName = "barber",
            .applicationVersion = comptime c.VK_MAKE_VERSION(0, 0, 1),
            .apiVersion = comptime c.VK_MAKE_VERSION(1, 2, 0),
        }),
        .enabledExtensionCount = instance_extensions[0..].len,
        .ppEnabledExtensionNames = &instance_extensions,
    });

    var instance: c.VkInstance = undefined;
    try vk.check(c.vkCreateInstance(&withLayerIfAvailable(temp_allocator, "VK_LAYER_KHRONOS_validation", instance_create_info), null, &instance));
    log.info("vulkan version 1.2 instance acquired", .{});

    const surface_create_info = vk.SType(c.VkWaylandSurfaceCreateInfoKHR, .{
        .display = @ptrCast(display),
        .surface = @ptrCast(surface),
    });
    var vk_surface: c.VkSurfaceKHR = undefined;
    try vk.check(c.vkCreateWaylandSurfaceKHR(instance, &surface_create_info, null, &vk_surface));
 
    const device = Device.select(temp_allocator, instance, vk_surface) orelse {
        log.err("failed to find suitable vulkan device. need version >=1.2 with graphics and bgra image format", .{});
        return error.NoSuitableVulkanDevice;
    };
    errdefer device.deinit();

    const swapchain = try Swapchain.init(safe_allocator, width, height, device, vk_surface);

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

    for (command_buffers, 0..) |command_buffer, i| {
        var begin_info = vk.SType(c.VkCommandBufferBeginInfo, .{});
        try vk.check(c.vkBeginCommandBuffer(command_buffer, &begin_info));
        c.vkCmdClearColorImage(command_buffer, swapchain.images[i], c.VK_IMAGE_LAYOUT_GENERAL, &.{ .float32 = .{0, 0, 0, 0} }, 1, &std.mem.zeroInit(c.VkImageSubresourceRange, .{
            .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        }));
        try vk.check(c.vkEndCommandBuffer(command_buffer));
    }

    var semaphore_info = vk.SType(c.VkSemaphoreCreateInfo, .{});
    var render_complete: c.VkSemaphore = undefined;
    var present_complete: c.VkSemaphore = undefined;
    try vk.check(c.vkCreateSemaphore(device.logical, &semaphore_info, null, &render_complete));
    try vk.check(c.vkCreateSemaphore(device.logical, &semaphore_info, null, &present_complete));

    // Compile shaders
    log.info("compiling shaders...", .{});

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

    log.info("finished compiling shaders.", .{});

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
    };
}

pub fn tick(context: *VulkanContext) !void {
    const gq = context.device.getGraphicsQueue();
    const pq = context.device.getPresentQueue();

    var image_index: u32 = undefined;
    try vk.check(c.vkAcquireNextImageKHR(context.device.logical, context.swapchain.handle, std.math.maxInt(u32), context.present_complete, null, &image_index));

    var submit_info = vk.SType(c.VkSubmitInfo, .{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &context.present_complete,
        .pWaitDstStageMask = &@as(u32, c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
        .commandBufferCount = 1,
        .pCommandBuffers = &context.command_buffers[image_index],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &context.render_complete,
    });
    try vk.check(c.vkQueueSubmit(gq, 1, &submit_info, null));

    var present_info = vk.SType(c.VkPresentInfoKHR, .{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &context.render_complete,
        .swapchainCount = 1,
        .pSwapchains = &context.swapchain.handle,
        .pImageIndices = &image_index,
    });
    try vk.check(c.vkQueuePresentKHR(pq, &present_info));
}

pub fn deinit(context: VulkanContext) void {
    c.vkDestroySemaphore(context.device.logical, context.render_complete, null);
    c.vkDestroySemaphore(context.device.logical, context.present_complete, null);
    c.vkFreeCommandBuffers(context.device.logical, context.command_pool, @intCast(context.command_buffers.len), context.command_buffers.ptr);
    c.vkDestroyCommandPool(context.device.logical, context.command_pool, null);
    context.swapchain.deinit(context);
    context.device.deinit();
}

// This is separated from the context directly so that in the future we can separate the lifetime with its own arena
const Swapchain = struct {
    arena: std.heap.ArenaAllocator, // for allocators that share the lifetime of the VulkanContext (usually static lifetime)
    handle: c.VkSwapchainKHR,
    images: []c.VkImage,
    views: []c.VkImageView,
    surface_caps: c.VkSurfaceCapabilitiesKHR,

    pub fn init(safe_allocator: std.mem.Allocator, width: u32, height: u32, device: Device, surface: c.VkSurfaceKHR) !Swapchain {
        var arena = std.heap.ArenaAllocator.init(safe_allocator);
        errdefer arena.deinit();

        var surface_caps: c.VkSurfaceCapabilitiesKHR = undefined;
        try vk.check(c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.physical, surface, &surface_caps));
        if ((surface_caps.supportedCompositeAlpha & c.VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) == 0) return error.AlphaChannelNotSupported;

        var swapchain_create_info = vk.SType(c.VkSwapchainCreateInfoKHR, .{
            .surface = surface,
            .minImageCount = surface_caps.minImageCount,
            .imageFormat = c.VK_FORMAT_R8G8B8A8_UNORM,
            .imageColorSpace = c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
            .imageExtent = .{ .width = width, .height = height },
            .imageArrayLayers = 1,
            .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | c.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
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

        var image_views = try arena.allocator().alloc(c.VkImageView, swapchain_images_count);
        for (swapchain_images, 0..) |image, i| {
            var image_view_create_info = vk.SType(c.VkImageViewCreateInfo, .{
                .image = image,
                .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
                .format = swapchain_create_info.imageFormat,
                .subresourceRange = .{
                    .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                    .levelCount = 1,
                    .layerCount = 1,
                },
            });
            try vk.check(c.vkCreateImageView(device.logical, &image_view_create_info, null, &image_views[i]));
        }
        image_views.len = swapchain_images_count;

        return .{
            .arena = arena,
            .handle = swapchain,
            .images = swapchain_images,
            .views = image_views,
            .surface_caps = surface_caps,
        };
    }

    pub fn deinit(swapchain: Swapchain, context: VulkanContext) void {
        for (swapchain.views) |view| {
            c.vkDestroyImageView(context.device.logical, view, null);
        }
        swapchain.arena.deinit();
        c.vkDestroySwapchainKHR(context.device.logical, swapchain.handle, null);
    }
};

const Device = struct {
    const QueuePair = struct {
        index: u32,
        value: c.VkQueue,
    };

    const QueueSetup = union(enum) {
        split: struct {
            present: QueuePair,
            graphics: QueuePair,
        },
        single: QueuePair, // single queue for graphics and present
    };

    logical: c.VkDevice,
    physical: c.VkPhysicalDevice,
    queue: QueueSetup,

    pub fn select(arena: std.mem.Allocator, instance: c.VkInstance, surface: c.VkSurfaceKHR) ?Device {
        var devices_count: u32 = 0;
        vk.check(c.vkEnumeratePhysicalDevices(instance, &devices_count, null)) catch unreachable;
        if (devices_count == 0) {
            return null;
        }
        var devices = arena.alloc(c.VkPhysicalDevice, devices_count) catch unreachable;
        vk.check(c.vkEnumeratePhysicalDevices(instance, &devices_count, devices.ptr)) catch unreachable;

        devices.len = devices_count;

        const priority_types = .{ c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 0 };
        inline for (priority_types) |prioritize_type| {
            for (devices) |device| {
                // Check the device extensions
                var extensions_count: u32 = 0;
                vk.check(c.vkEnumerateDeviceExtensionProperties(device, null, &extensions_count, null)) catch unreachable;
                if (extensions_count == 0) continue;

                var extensions = arena.alloc(c.VkExtensionProperties, extensions_count) catch unreachable;
                vk.check(c.vkEnumerateDeviceExtensionProperties(device, null, &extensions_count, extensions.ptr)) catch unreachable;
                extensions.len = extensions_count;

                var swapchain_support = false;
                for (extensions) |extension| {
                    const extension_name = "VK_KHR_swapchain";
                    if (std.mem.orderZ(u8, "VK_KHR_swapchain", @ptrCast(extension.extensionName[0..extension_name.len])) == .eq) {
                        swapchain_support = true;
                    }
                }
                if (!swapchain_support) continue;

                // Check the device properties
                var props: c.VkPhysicalDeviceProperties = undefined;
                c.vkGetPhysicalDeviceProperties(device, &props);
                if (prioritize_type != 0 and props.deviceType != prioritize_type) continue;

                // Check the features
                var features: c.VkPhysicalDeviceFeatures = undefined;
                c.vkGetPhysicalDeviceFeatures(device, &features);
                if (features.geometryShader == c.VK_FALSE) continue;

                // Check the queue families
                var queue_family_count: u32 = 0;
                c.vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, null);
                if (queue_family_count == 0) {
                    continue;
                }

                var queue_families = arena.alloc(c.VkQueueFamilyProperties, queue_family_count) catch unreachable;
                c.vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.ptr);
                queue_families.len = queue_family_count;

                var present_queue_index: ?usize = null;
                var graphics_queue_index: ?usize = null;

                for (queue_families, 0..) |qf, qi| {
                    if ((qf.queueFlags & c.VK_QUEUE_GRAPHICS_BIT) != 0) {
                        graphics_queue_index = qi;
                    }
                    var surface_support = c.VK_FALSE;
                    vk.check(c.vkGetPhysicalDeviceSurfaceSupportKHR(device, @intCast(qi), surface, &surface_support)) catch unreachable;
                    if (surface_support == c.VK_TRUE) {
                        present_queue_index = qi;
                    }
                }
                if (graphics_queue_index == null or present_queue_index == null) continue;

                // We don't need to check for surface format support right now
                // as we are using one of the mandatory formats VK_FORMAT_R8G8B8A8_UNORM
                var formats_count: u32 = 0;
                vk.check(c.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formats_count, null)) catch unreachable;
                if (formats_count == 0) continue;

                var present_modes_count: u32 = 0;
                vk.check(c.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_modes_count, null)) catch unreachable;
                if (present_modes_count == 0) continue;

                // Device is suitable, setup logical device and fetch the graphics queue
 
                var device_create_info = vk.SType(c.VkDeviceCreateInfo, .{
                    .queueCreateInfoCount = 1,
                    .pQueueCreateInfos = &vk.SType(c.VkDeviceQueueCreateInfo, .{
                        .queueCount = 1, // we are only submitting to one queue even if more are available
                        .pQueuePriorities = &@as(f32, 0.0),
                        .queueFamilyIndex = @intCast(graphics_queue_index.?),
                    }),
                    .enabledExtensionCount = 1,
                    .ppEnabledExtensionNames = &"VK_KHR_swapchain".ptr,
                });
                var logical_device: c.VkDevice = undefined;
                vk.check(c.vkCreateDevice(device, &device_create_info, null, &logical_device)) catch unreachable;

                var queue: QueueSetup = undefined;
                if (graphics_queue_index.? == present_queue_index.?) {
                    var graphics_queue: c.VkQueue = undefined;
                    c.vkGetDeviceQueue(logical_device, @intCast(graphics_queue_index.?), 0, &graphics_queue);
                    queue = .{
                        .single = .{
                            .index = @intCast(graphics_queue_index.?),
                            .value = graphics_queue,
                        }
                    };
                } else {
                    var present_queue: c.VkQueue = undefined;
                    c.vkGetDeviceQueue(logical_device, @intCast(present_queue_index.?), 0, &present_queue);
                    var graphics_queue: c.VkQueue = undefined;
                    c.vkGetDeviceQueue(logical_device, @intCast(graphics_queue_index.?), 0, &graphics_queue);
                    queue = .{
                        .split = .{
                            .present = .{
                                .index = @intCast(present_queue_index.?),
                                .value = present_queue,
                            },
                            .graphics = .{
                                .index = @intCast(graphics_queue_index.?),
                                .value = graphics_queue,
                            },
                        },
                    };
                }

                log.info("selected vulkan device '{s}'", .{props.deviceName});

                return .{
                    .physical = device,
                    .logical = logical_device,
                    .queue = queue,
                };
            }
        }

        return null;
    }

    pub fn getGraphicsQueueIndex(device: Device) u32 {
        switch (device.queue) {
            .single => |queue| return queue.index,
            .split => |queue_pair| return queue_pair.graphics.index,
        }
    }

    pub fn getGraphicsQueue(device: Device) c.VkQueue {
        switch (device.queue) {
            .single => |queue| return queue.value,
            .split => |queue_pair| return queue_pair.graphics.value,
        }
    }

    pub fn getPresentQueue(device: Device) c.VkQueue {
        switch (device.queue) {
            .single => |queue| return queue.value,
            .split => |queue_pair| return queue_pair.present.value,
        }
    }

    pub fn deinit(device: Device) void {
        c.vkDestroyDevice(device.logical, null);
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
