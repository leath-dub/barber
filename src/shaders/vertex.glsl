#version 460

#extension GL_EXT_debug_printf : enable

layout(location = 0) in vec3 in_Position;
layout(location = 1) in vec2 in_Uv;

vec2 pos[3] = vec2[3]( vec2(-0.7, 0.7), vec2(0.7, 0.7), vec2(0.0, -0.7) );

void main() {
  gl_Position = vec4(in_Position, 1.0);
}
