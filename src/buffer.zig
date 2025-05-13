const std = @import("std");

const vk = @import("vk.zig");
const Device = @import("device.zig");

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

// pub const Buffer = struct {
//     handle: c.VkBuffer,
//     memory: c.VkMemory,
//     capacity: usize,
//
//     pub fn init(device: Device, capacity: usize, usage: c.VkBufferUsageFlags, memory_props: c.VkMemoryPropertyFlags) !Buffer {
//         var result: Buffer = undefined;
//
//         var buffer_info = vk.SType(c.VkBufferCreateInfo, .{
//             .size = capacity,
//             .usage = usage,
//             .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
//         });
//         try vk.check(c.vkCreateBuffer(device.logical, &buffer_info, null, &result.handle));
//
//         var memory_reqs: c.VkMemoryRequirements  = undefined;
//         c.vkGetBufferMemoryRequirements(device.logical, buffer, &memory_reqs);
//
//         result.capacity = memory_reqs.size;
//
//         const Set = std.bit_set.IntegerBitSet(32);
//         var type_bits = Set { .mask = memory_reqs.memoryTypeBits };
//         var prop_bits = Set { .mask = memory_props };
//         const memory_types = device.memory_props.memoryTypes[0..device.memory_props.memoryTypeCount];
//         const type_index = for (memory_types, 0..) |memory_type, i| {
//             if (type_bits.isSet(i) and prop_bits.eql(.{ .mask = memory_type.propertyFlags })) {
//                 break i;
//             }
//         };
//
//         var alloc_info = vk.SType(c.VkMemoryAllocateInfo, .{
//             .allocationSize = capacity,
//             .memoryTypeIndex = type_index,
//         });
//         try vk.check(c.vkAllocateMemory(device, &alloc_info, null, &result.memory));
//         try vk.check(c.vkBindBufferMemory(device, buffer, result.memory, 0));
//
//         return result;
//     }
//
//     pub fn ensureCapacity(buf: *Buffer, capacity: usize) !void {
//         if (capacity > buf.capacity) {
//             c.
//         }
//     }
//
//     pub fn deinit(device: c.VkDevice, buf: Buffer) void {
//         c.vkDestroyBuffer(device, buf.handle, null);
//         c.vkFreeMemory(device, buf.memory, null);
//     }
// };

// pub fn HandlePool(comptime T: type) type {
//     return struct {
//         const Pool = @This();
//         const Handle = usize;
//         const Node = struct {
//             block: Handle,
//             next: ?*Node,
//         };
//
//         const Block = union(enum) {
//             vacant: void,
//             uninit: void,
//             init: T,
//         };
//
//         arena: std.heap.ArenaAllocator,
//         blocks: []Block,
//         capacity: usize = 0,
//         free_list: ?*Node,
//
//         pub fn initCapacity(allocator: std.mem.Allocator, num: usize) !Pool {
//             if (num == 0) return error.InvalidSize;
//
//             var result: Pool = undefined;
//
//             result.arena = std.heap.ArenaAllocator.init(allocator);
//             result.blocks = try result.arena.allocator().alloc(T, num);
//             @memset(result.blocks, .vacant);
//             result.capacity = num;
//             result.free_list = try result.arena.allocator().create(Node);
//             result.free_list.?.block = 0;
//             result.free_list.?.next = null;
//
//             result.updateFreeList(0);
//
//             return result;
//         }
//
//         fn updateFreeList(pool: *Pool, old_capacity: usize) !void {
//             // find the end of the free list
//             var node_ptr = pool.free_list;
//             while (node_ptr.next != null) node_ptr = node_ptr.?.next;
//
//             for (old_capacity..pool.capacity) |i| {
//                 node_ptr.next = try pool.arena.allocator().create(Node);
//                 node_ptr.next.block = i;
//                 node_ptr = node_ptr.next;
//                 node_ptr.next = null;
//             }
//         }
//
//         pub fn create(pool: *Pool) Handle {
//             if (pool.free_list) |free_list| {
//                 const block = free_list.block;
//                 pool.free_list = free_list.next;
//                 pool.arena.allocator().destroy(free_list);
//                 return block;
//             }
//
//             const block = pool.capacity;
//             pool.capacity *= 2;
//             pool.blocks = try pool.arena.allocator().realloc(pool.blocks, pool.capacity);
//             @memset(pool.blocks[block..], .vacant);
//             try pool.updateFreeList(block);
//
//             return block;
//         }
//
//         pub fn destroy(pool: *Pool, block: Handle) !void {
//             var new_node = try pool.arena.allocator().create(Node);
//             new_node.block = block;
//             new_node.next = pool.free_list;
//             pool.free_list = new_node;
//         }
//
//         pub fn get(pool: *Pool, block: Handle) *Block {
//             return &pool.blocks[block];
//         }
//
//         pub fn deinit(pool: Pool) void {
//             pool.arena.deinit();
//         }
//     };
// }
