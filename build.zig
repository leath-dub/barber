const std = @import("std");
const Build = std.Build;

const Scanner = @import("wayland").Scanner;

pub fn build(b: *Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const scanner = Scanner.create(b, .{});
    const wayland = b.createModule(.{ .root_source_file = scanner.result });
    
    scanner.addSystemProtocol("stable/xdg-shell/xdg-shell.xml"); // needed by wlr-layer-shell
    scanner.addCustomProtocol(b.path("wlr-protocols/unstable/wlr-layer-shell-unstable-v1.xml"));

    scanner.generate("wl_output", 2);
    scanner.generate("wl_compositor", 1);
    scanner.generate("zwlr_layer_shell_v1", 1);

    const exe = b.addExecutable(.{
        .name = "barber",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const zmesh = b.dependency("zmesh", .{ .target = target });
    exe.root_module.addImport("zmesh", zmesh.module("root"));
    exe.linkLibrary(zmesh.artifact("zmesh"));

    const zmath = b.dependency("zmath", .{ .target = target });
    exe.root_module.addImport("zmath", zmath.module("root"));

    exe.root_module.addImport("wayland", wayland);
    exe.linkLibC();
    exe.linkSystemLibrary("wayland-client");
    exe.linkSystemLibrary("vulkan");
 
    // Shader compilation is done at runtime for more flexiblity (maybe custom shaders in the future?) using glslang
    exe.linkSystemLibrary("glslang");
    exe.linkSystemLibrary("glslang-default-resource-limits");

    b.installArtifact(exe);

    const run_exe = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_exe.step);
}
