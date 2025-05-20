#version 460

#extension GL_EXT_debug_printf : enable

layout(push_constant) uniform Push {
  float aabb[6];
} push;

layout(location = 0) in vec3 in_Position;
layout(location = 1) in vec3 in_Normal;
layout(location = 2) in vec2 in_Texcoord;

void main() {
  gl_Position = vec4(in_Position - vec3(0, 0, -0.2), 1.0);
}
