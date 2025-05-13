const std = @import("std");
const builtin = std.builtin;
const meta = std.meta;
const mem = std.mem;
const heap = std.heap;
const enums = std.enums;

const vk = @import("vk.zig");
const Device = @import("device.zig");
const Memory = @import("buffer.zig").Memory;

const c = @import("c.zig").includes;

pub fn Vec(cardinality: comptime_int) type {
    return @Vector(cardinality, f32);
}

pub const Vert = struct {
    position: Vec(3),
    texel: Vec(2),
    pub fn of(comptime components: anytype) Vert {
        var result: Vert = undefined;
        inline for (comptime meta.fieldNames(@TypeOf(components)), 0..) |field_name, i| {
            @field(result, @typeInfo(Vert).@"struct".fields[i].name) = @field(components, field_name);
        }
        return result;
    }
};

pub fn SoA(comptime T: type) type {
    if (@typeInfo(T) != .@"struct") @compileError("SoA can only works for structs");
    const fields = meta.fields(T);
    if (fields.len == 0) @compileError("SoA input type must have fields");
    var new_fields: [fields.len]builtin.Type.StructField = undefined;
    inline for (fields, 0..) |field, i| {
        if (field.is_comptime) @compileError("SoA cannot hold comptime fields");
        var new_field = field;
        new_field.type = @Type(.{
            .pointer = .{
                .size = .slice,
                .is_const = false,
                .is_volatile = false,
                .alignment = @alignOf(field.type),
                .address_space = .generic,
                .child = field.type,
                .is_allowzero = false,
                .sentinel_ptr = null,
            },
        });
        new_fields[i] = new_field;
    }
    return @Type(.{
        .@"struct" = .{
            .layout = .auto,
            .fields = &new_fields,
            .decls = &.{},
            .is_tuple = false,
        }
    });
}

pub const Verts = struct {
    soa: SoA(Vert),
    arena: heap.ArenaAllocator,

    pub fn init(allocator: mem.Allocator, vert_count: usize) !Verts {
        var result: Verts = undefined;
        result.arena = heap.ArenaAllocator.init(allocator);
        inline for (meta.fields(Vert)) |field| {
            @field(result.soa, field.name) = try result.arena.allocator().alloc(field.type, vert_count);
        }
        return result;
    }

    pub fn with(allocator: mem.Allocator, data: []const Vert) !Verts {
        var verts = try Verts.init(allocator, data.len);
        for (data, 0..) |vert, i| {
            inline for (comptime meta.fieldNames(Vert)) |field_name| {
                verts.refMut(meta.stringToEnum(meta.FieldEnum(Vert), field_name).?)[i] = @field(vert, field_name);
            }
        }
        return verts;
    }

    pub fn ref(verts: *const Verts, comptime field: meta.FieldEnum(Vert)) []const @FieldType(Vert, @tagName(field)) {
        return @field(verts.soa, @tagName(field));
    }

    pub fn refMut(verts: *Verts, comptime field: meta.FieldEnum(Vert)) []@FieldType(Vert, @tagName(field)) {
        return @field(verts.soa, @tagName(field));
    }

    pub fn len(verts: Verts) usize {
        // Arbitrarily pick the first field to check the length
        return @field(verts.soa, @typeInfo(Vert).@"struct".fields[0].name).len;
    }

    pub fn deinit(verts: Verts) void {
        verts.arena.deinit();
    }
};

fn maxField(comptime T: type) usize {
    var max: usize = 0;
    inline for (meta.fields(T)) |field| {
        if (@sizeOf(field.type) > max) {
            max = @sizeOf(field.type);
        }
    }
    return max;
}

pub const Mesh = struct {
    device: Device,
    buffers: enums.EnumArray(meta.FieldEnum(Vert), c.VkBuffer),
    memory: Memory,
    buffer_offsets: enums.EnumArray(meta.FieldEnum(Vert), usize),
    size: usize,

    pub fn init(device: Device, copy_cmdbuf: c.VkCommandBuffer, verts: Verts) !Mesh {
        // The final buffers
        var buffers = enums.EnumArray(meta.FieldEnum(Vert), c.VkBuffer).initUndefined();
        var buffer_offsets = enums.EnumArray(meta.FieldEnum(Vert), usize).initUndefined();
        inline for (comptime meta.tags(meta.FieldEnum(Vert))) |attachment| {
            const field_size = @sizeOf(@FieldType(Vert, @tagName(attachment)));
            var buffer_info = vk.SType(c.VkBufferCreateInfo, .{
                .size = verts.len() * field_size,
                .usage = c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            });
            try vk.check(c.vkCreateBuffer(device.logical, &buffer_info, null, buffers.getPtr(attachment)));
        }
 
        // Even though we store vertex data as a struct of arrays, we only allocate a fixed chunk of memory.
        // We just use sub binding of buffers onto the allocated block to get the appearence of completely separate
        // arrays.
        var final_buffer_reqs: c.VkMemoryRequirements  = undefined;
        c.vkGetBufferMemoryRequirements(device.logical, buffers.get(.position), &final_buffer_reqs);
        const mesh_memory = try Memory.fromBufs(device, &buffers.values, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        var staging_buffer: c.VkBuffer = undefined;
        var buffer_info = vk.SType(c.VkBufferCreateInfo, .{
            .size = verts.len() * maxField(Vert),
            .usage = c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        });
        try vk.check(c.vkCreateBuffer(device.logical, &buffer_info, null, &staging_buffer));

        var memory_reqs: c.VkMemoryRequirements  = undefined;
        c.vkGetBufferMemoryRequirements(device.logical, staging_buffer, &memory_reqs);

        var staging_memory = try Memory.init(device, verts.len() * maxField(Vert), memory_reqs.memoryTypeBits, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        const buffer_offset = try staging_memory.bind(staging_buffer);
        defer staging_memory.deinit();
        std.debug.assert(buffer_offset == 0);

        inline for (comptime meta.tags(meta.FieldEnum(Vert))) |attachment| {
            const data = verts.ref(attachment);
            const stage = try staging_memory.map(data.len * @sizeOf(@typeInfo(@TypeOf(data)).pointer.child));
            @memcpy(stage, @as([*]const u8, @ptrCast(data))[0..stage.len]);
            staging_memory.unmap();

            {
                var begin_info = vk.SType(c.VkCommandBufferBeginInfo, .{ .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT });
                try vk.check(c.vkBeginCommandBuffer(copy_cmdbuf, &begin_info));

                c.vkCmdCopyBuffer(copy_cmdbuf, staging_buffer, buffers.get(attachment), 1, &c.VkBufferCopy {
                    .srcOffset = buffer_offset,
                    .dstOffset = buffer_offsets.get(attachment),
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

        return .{
            .device = device,
            .buffers = buffers,
            .memory = mesh_memory,
            .buffer_offsets = buffer_offsets,
            .size = verts.len(),
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
