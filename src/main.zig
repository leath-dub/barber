const std = @import("std");
const log = std.log;
const posix = std.posix;
const mem = std.mem;

const wayland = @import("wayland");
const wl = wayland.client.wl;
const zwlr = wayland.client.zwlr;

const Backend = @import("backend.zig");

const zmesh = @import("zmesh");

const c = @import("c.zig").includes;

pub fn abort(comptime format: []const u8, args: anytype) noreturn {
    log.err(format, args);
    posix.exit(1);
}

const Context = struct {
    safe_allocator: std.mem.Allocator,
    frame_arena: std.mem.Allocator,
    display: *wl.Display,
    compositor: ?*wl.Compositor = null,
    layer_shell: ?*zwlr.LayerShellV1 = null,
    output: ?*wl.Output = null,
    surface: ?*wl.Surface = null,
    backend: ?Backend = null,
};

fn registryListener(registry: *wl.Registry, event: wl.Registry.Event, context: *Context) void {
    switch (event) {
        .global => |global| {
            if (mem.orderZ(u8, global.interface, wl.Compositor.interface.name) == .eq) {
                context.compositor = registry.bind(global.name, wl.Compositor, 1) catch return;
            } else if (mem.orderZ(u8, global.interface, wl.Output.interface.name) == .eq) {
                context.output = registry.bind(global.name, wl.Output, 2) catch return;
            } else if (mem.orderZ(u8, global.interface, zwlr.LayerShellV1.interface.name) == .eq) {
                context.layer_shell = registry.bind(global.name, zwlr.LayerShellV1, global.version) catch return;
            }
        },
        .global_remove => {},
    }
}

fn layerSurfaceListener(layer_surface: *zwlr.LayerSurfaceV1, event: zwlr.LayerSurfaceV1.Event, context: *Context) void {
    switch (event) {
        .configure => |configure| {
            log.info("Configure: height: {d}, width: {d}, serial: {d}", .{configure.height, configure.width, configure.serial});
            if (context.backend == null) {
                context.backend = Backend.init(context.safe_allocator, context.frame_arena, configure.width, configure.height, context.surface.?, context.display) catch unreachable;
                layer_surface.ackConfigure(configure.serial);
            } else {
                // TODO is it ok to ignore this ??
                log.warn("another configure event raised: height: {d}, width: {d}, serial: {d}", .{configure.height, configure.width, configure.serial});
                layer_surface.ackConfigure(configure.serial);
            }
        },
        .closed => {},
    }
}

pub fn main() !void {
    var safe_allocator = std.heap.DebugAllocator(.{}).init;
    defer {
        if (safe_allocator.deinit() == .leak) {
            _ = safe_allocator.detectLeaks();
            @panic("Memory leaks detected");
        }
    }
    zmesh.init(safe_allocator.allocator());
    defer zmesh.deinit();

    var arena_allocator = std.heap.ArenaAllocator.init(safe_allocator.allocator());
    defer arena_allocator.deinit();
    const arena = arena_allocator.allocator();

    const display = wl.Display.connect(null) catch abort("failed to connect to wayland display", .{});
    defer display.disconnect();
    log.info("connection to wayland display established", .{});

    const registry = try display.getRegistry();

    var context: Context = .{ .safe_allocator = safe_allocator.allocator(), .frame_arena = arena, .display = display };
    registry.setListener(*Context, registryListener, &context);
    if (display.roundtrip() != .SUCCESS) abort("round trip failed", .{});
    defer context.backend.?.deinit();

    const compositor = context.compositor orelse abort("failed to acquire wl_compositor interface", .{});
    const surface = compositor.createSurface() catch abort("failed to create wayland surface", .{});
    context.surface = surface;

    const layer_shell = context.layer_shell orelse abort("failed to aquire zwlr_layer_shell_v1 interface - does your compositor support it?", .{});
    const output = context.output orelse abort("failed to aquire wl_output interface", .{});

    const layer_surface = try layer_shell.getLayerSurface(surface, output, .overlay, "barber");

    // Initial configuration of the status bar
    layer_surface.setAnchor(.{ .top = true, .left = true, .right = true, .bottom = false });
    layer_surface.setExclusiveZone(50);
    // layer_surface.setMargin(10, 10, 10, 10);
    layer_surface.setSize(0, 50);
    surface.commit();

    layer_surface.setListener(*Context, layerSurfaceListener, &context);

    while (display.dispatch() == .SUCCESS) {
        try context.backend.?.tick();
        _ = display.flush();
        _ = arena_allocator.reset(.{ .retain_with_limit = 8192 });
    }
}
