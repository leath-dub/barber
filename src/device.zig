const std = @import("std");
const log = std.log;

const c = @import("c.zig").includes;

const vk = @import("vk.zig");

const Device = @This();

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
memory_props: c.VkPhysicalDeviceMemoryProperties,

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

            var memory_props: c.VkPhysicalDeviceMemoryProperties = undefined;
            c.vkGetPhysicalDeviceMemoryProperties(device, &memory_props);

            return .{
                .physical = device,
                .logical = logical_device,
                .queue = queue,
                .memory_props = memory_props,
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
