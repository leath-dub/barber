const std = @import("std");
const builtin = std.builtin;
const meta = std.meta;
const mem = std.mem;
const heap = std.heap;
const enums = std.enums;
const log = std.log;

const vk = @import("vk.zig");
const Device = @import("device.zig");
const Memory = @import("buffer.zig").Memory;

const zmesh = @import("zmesh");

const c = @import("c.zig").includes;

fn VertexBufferMap(comptime V: type) type {
    return struct {
        const Self = @This();

        values: [meta.tags(VertexBuffer.Index).len]V,

        pub fn get(bufs: Self, index: VertexBuffer.Index) V {
            return bufs.values[index.ord()];
        }

        pub fn getPtr(bufs: *Self, index: VertexBuffer.Index) *V {
            return &bufs.values[index.ord()];
        }

        const Entry = struct {
            key: VertexBuffer.Index,
            value: *const V,
        };

        const EntryIterator = struct {
            bufs: *const Self,
            index: usize = 0,
            pub fn next(it: *EntryIterator) ?Entry {
                if (it.index >= it.bufs.values.len) return null;
                defer it.index += 1;
                return .{ .key = VertexBuffer.Index.at(it.index), .value = &it.bufs.values[it.index] };
            }
        };

        pub fn iterator(bufs: *const Self) EntryIterator {
            return .{ .bufs = bufs };
        }
    };
}

pub const VertexBuffer = struct {
    pub const Index = enum(usize) {
        indices = 0,
        positions,
        normals,
        texcoords,

        pub fn ord(index: Index) usize {
            return @intFromEnum(index);
        }

        pub fn at(index: usize) Index {
            return @enumFromInt(index);
        }
    };

    indices: []u32,
    positions: [][3]f32,
    normals: ?[][3]f32,
    texcoords: ?[][2]f32,

    const Field = meta.FieldEnum(VertexBuffer);
    pub const Bindings = [_]Index{ .positions, .normals, .texcoords };
    pub const binding_type = .{
        .positions = c.VK_FORMAT_R32G32B32_SFLOAT,
        .normals = c.VK_FORMAT_R32G32B32_SFLOAT,
        .texcoords = c.VK_FORMAT_R32G32_SFLOAT,
    };

    pub fn sizeOf(comptime field: VertexBuffer.Index) usize {
        return @sizeOf(VertexBuffer.typeOf(field));
    }

    pub fn typeOf(comptime field: VertexBuffer.Index) type {
        const info = @typeInfo(@FieldType(VertexBuffer, @tagName(field)));
        return (if (info == .optional)
            @typeInfo(info.optional.child)
        else info).pointer.child;
    }

    pub fn isOptional(comptime field: VertexBuffer.Index) bool {
        return @typeInfo(@FieldType(VertexBuffer, @tagName(field))) == .optional;
    }

    pub fn len(vb: VertexBuffer, comptime field: VertexBuffer.Index) usize {
        const items = @field(vb, @tagName(field));
        if (@typeInfo(@TypeOf(items)) == .optional)
            return items.?.len
        else return items.len;
    }

    pub fn count(vb: VertexBuffer) usize {
        return vb.positions.len;
    }
};

fn flatMapOpt(x: anytype) t: {
    const info = @typeInfo(@TypeOf(x));
    if (info == .optional) break :t @TypeOf(x)
    else break :t ?@TypeOf(x);
} {
    return x;
}

pub const Mesh = struct {
    const Buffers = VertexBufferMap(c.VkBuffer);
    const BufferSizes = VertexBufferMap(usize);

    device: Device,
    memory: Memory,
    buffers: Buffers,
    sizes: BufferSizes,

    pub fn init(device: Device, copy_cmdbuf: c.VkCommandBuffer, vb: VertexBuffer) !Mesh {
        // The final buffers
        var buffers: Buffers = undefined;
        var buffer_sizes: BufferSizes = undefined;
        inline for (comptime meta.tags(VertexBuffer.Index)) |attachment| {
            var size: c.VkDeviceSize = undefined;
            var usage: u32 = c.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            if (attachment == .indices) {
                usage |= c.VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
                size = vb.len(.indices) * VertexBuffer.sizeOf(attachment);
            } else {
                usage |= c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
                size = vb.count() * VertexBuffer.sizeOf(attachment);
            }
            var buffer_info = vk.SType(c.VkBufferCreateInfo, .{
                .size = size,
                .usage = usage,
                .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            });
            try vk.check(c.vkCreateBuffer(device.logical, &buffer_info, null, buffers.getPtr(attachment)));
            buffer_sizes.getPtr(attachment).* = size;
        }
 
        // Even though we store vertex data as a struct of arrays, we only allocate a fixed chunk of memory.
        // We just use sub binding of buffers onto the allocated block to get the appearence of completely separate
        // arrays.
        var final_buffer_reqs: c.VkMemoryRequirements  = undefined;
        c.vkGetBufferMemoryRequirements(device.logical, buffers.get(.positions), &final_buffer_reqs);
        const mesh_memory = try Memory.fromBufs(device, &buffers.values, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        var staging_buffer: c.VkBuffer = undefined;
        var buffer_info = vk.SType(c.VkBufferCreateInfo, .{
            .size = mesh_memory.capacity,
            .usage = c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        });
        try vk.check(c.vkCreateBuffer(device.logical, &buffer_info, null, &staging_buffer));

        var memory_reqs: c.VkMemoryRequirements  = undefined;
        c.vkGetBufferMemoryRequirements(device.logical, staging_buffer, &memory_reqs);

        var staging_memory = try Memory.init(device, mesh_memory.capacity, memory_reqs.memoryTypeBits, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        const buffer_offset = try staging_memory.bind(staging_buffer);
        defer staging_memory.deinit();
        std.debug.assert(buffer_offset == 0);

        inline for (comptime meta.tags(VertexBuffer.Index)) |attachment| {
            const data_opt = flatMapOpt(@field(vb, @tagName(attachment)));
            if (data_opt) |data| {
                const stage = try staging_memory.map(data.len * VertexBuffer.sizeOf(attachment));
                @memcpy(stage, @as([*]const u8, @ptrCast(data))[0..stage.len]);
                staging_memory.unmap();

                {
                    var begin_info = vk.SType(c.VkCommandBufferBeginInfo, .{ .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT });
                    try vk.check(c.vkBeginCommandBuffer(copy_cmdbuf, &begin_info));

                    c.vkCmdCopyBuffer(copy_cmdbuf, staging_buffer, buffers.get(attachment), 1, &c.VkBufferCopy {
                        .srcOffset = buffer_offset,
                        .dstOffset = 0,
                        .size = stage.len,
                    });

                    try vk.check(c.vkEndCommandBuffer(copy_cmdbuf));
                }

                const gq = device.getGraphicsQueue();
                var submit_info = vk.SType(c.VkSubmitInfo, .{
                    .commandBufferCount = 1,
                    .pCommandBuffers = &copy_cmdbuf,
                });
                try vk.check(c.vkQueueSubmit(gq, 1, &submit_info, null));
                try vk.check(c.vkQueueWaitIdle(gq));
            }
        }

        return .{
            .device = device,
            .memory = mesh_memory,
            .buffers = buffers,
            .sizes = buffer_sizes,
        };
    }

    pub fn deinit(mesh: Mesh) void {
        var buffers = mesh.buffers;
        var it = buffers.iterator();
        while (it.next()) |buffer| {
            c.vkDestroyBuffer(mesh.device.logical, buffer.value.*, null);
        }
        mesh.memory.deinit();
    }
};
