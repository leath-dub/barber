const std = @import("std");

const vk = @import("vk.zig");
const Device = @import("device.zig");

const zm = @import("zmath");

const c = @import("c.zig").includes;

pub const Memory = struct {
    device: Device,
    memory: c.VkDeviceMemory,
    used: usize,
    capacity: usize,

    pub fn init(device: Device, capacity: usize, memory_type_bits: u32, props: c.VkMemoryPropertyFlags) !Memory {
        var alloc_info = vk.SType(c.VkMemoryAllocateInfo, .{
            .allocationSize = capacity,
            .memoryTypeIndex = getMemoryTypeIndex(device, memory_type_bits, props).?,
        });
        var memory: c.VkDeviceMemory = undefined;
        try vk.check(c.vkAllocateMemory(device.logical, &alloc_info, null, &memory));
        return .{
            .device = device,
            .memory = memory,
            .used = 0,
            .capacity = capacity,
        };
    }

    pub fn fromBufs(device: Device, bufs: []c.VkBuffer, props: c.VkMemoryPropertyFlags) !Memory {
        var total_size: usize = 0;
        var memory_type_bits: u32 = 0;

        for (bufs) |buf| {
            var reqs: c.VkMemoryRequirements = undefined;
            c.vkGetBufferMemoryRequirements(device.logical, buf, &reqs);
            memory_type_bits |= reqs.memoryTypeBits;
            const padding = (reqs.alignment - (total_size % reqs.alignment)) % reqs.alignment;
            total_size += padding + reqs.size;
        }

        var mem = try Memory.init(device, total_size, memory_type_bits, props);

        for (bufs) |buf| {
            _ = try mem.bind(buf);
        }

        return mem;
    }

    pub fn bind(mem: *Memory, buffer: c.VkBuffer) !usize {
        var reqs: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(mem.device.logical, buffer, &reqs);

        // Ensure new offset is aligned
        const padding = (reqs.alignment - (mem.used % reqs.alignment)) % reqs.alignment;
        const new_offset = mem.used + padding;

        if (mem.capacity - new_offset < reqs.size) return error.BufferTooLarge;
        try vk.check(c.vkBindBufferMemory(mem.device.logical, buffer, mem.memory, new_offset));

        mem.used = new_offset + reqs.size;
        return new_offset;
    }

    pub fn map(mem: Memory, size: usize) ![]u8 {
        var data: []u8 = undefined;
        data.len = size;
        try vk.check(c.vkMapMemory(mem.device.logical, mem.memory, 0, size, 0, @ptrCast(&data.ptr)));
        return data;
    }

    pub fn unmap(mem: Memory) void {
        c.vkUnmapMemory(mem.device.logical, mem.memory);
    }

    // This could be more optimal by re arranging the order, but that is over complicated!
    pub fn calculateSize(allocs: []const usize, alignment: usize) usize {
        var capacity: usize = 0;
        for (allocs, 0..) |alloc, i| {
            const padding = (alignment[i] - (capacity % alignment[i])) % alignment[i];
            const new_offset = capacity + padding;
            capacity = new_offset + alloc;
        }
        return capacity;
    }

    pub fn reset(mem: *Memory) void {
        mem.used = 0;
    }

    pub fn deinit(mem: Memory) void {
        c.vkFreeMemory(mem.device.logical, mem.memory, null);
    }

    fn getMemoryTypeIndex(device: Device, memory_type_bits: u32, props: c.VkMemoryPropertyFlags) ?u32 {
        const Set = std.bit_set.IntegerBitSet(32);
        var type_bits = Set { .mask = memory_type_bits };
        var prop_bits = Set { .mask = props };
        const memory_types = device.memory_props.memoryTypes[0..device.memory_props.memoryTypeCount];
        for (memory_types, 0..) |memory_type, i| {
            if (type_bits.isSet(i) and prop_bits.subsetOf(.{ .mask = memory_type.propertyFlags })) {
                return @intCast(i);
            }
        }
        return null;
    }
};


pub const UniformBuffer = struct {
    pub const Data = struct {
        model: zm.Mat,
        view: zm.Mat,
        projection: zm.Mat,

        pub fn default(aspect: f32) @This() {
            // Convert 0..1 to -1..1 range
            const model = zm.transpose(zm.Mat {
                zm.f32x4(2, 0, 0, -1),
                zm.f32x4(0, 2, 0, -1),
                zm.f32x4(0, 0, 1,  0),
                zm.f32x4(0, 0, 0,  1),
            });
            const view = zm.lookAtLh(
                zm.f32x4(0.0, 0.0, -2, 0.0),
                zm.f32x4(0.0, 0.0, 0.0, 0.0),
                zm.f32x4(0.0, 1.0, 0.0, 0.0),
            );
            const projection = zm.perspectiveFovLh(45, aspect, 0.1, 1000.0);
            return .{ .model = model, .view = view, .projection = projection };
        }
    };

    device: Device,
    buffer: c.VkBuffer,
    memory: Memory,

    pub fn init(device: Device) !UniformBuffer {
        var buffer_info = vk.SType(c.VkBufferCreateInfo, .{
            .size = @sizeOf(Data),
            .usage = c.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        });
        var buffer: c.VkBuffer = undefined;
        try vk.check(c.vkCreateBuffer(device.logical, &buffer_info, null, &buffer));
        const memory = try Memory.fromBufs(device, @as([*]c.VkBuffer, @ptrCast(&buffer))[0..1], c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        return .{
            .device = device,
            .buffer = buffer,
            .memory = memory,
        };
    }

    pub fn update(ub: UniformBuffer, data: Data) !void {
        const view = try ub.memory.map(@sizeOf(Data));
        defer ub.memory.unmap();
        @memcpy(view, @as([*]const u8, @ptrCast(&data))[0..view.len]);
    }

    pub fn deinit(ub: UniformBuffer) void {
        ub.memory.deinit();
        c.vkDestroyBuffer(ub.device.logical, ub.buffer, null);
    }
};
