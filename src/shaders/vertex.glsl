#version 460

#extension GL_EXT_debug_printf : enable

layout(location = 0) in vec3 in_Position;
layout(location = 1) in vec3 in_Normal;
layout(location = 2) in vec2 in_Texcoord;

layout(set = 0, binding = 0) uniform UniformBuffer {
  mat4 model;
  mat4 view;
  mat4 projection;
} ub;

void main() {
  gl_Position = ub.projection * ub.view * ub.model * vec4(in_Position, 1.0);
}
