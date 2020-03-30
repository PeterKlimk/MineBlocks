#version 450

layout (push_constant) uniform PushConstantData
{
    mat4 projection;
} pc;

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 tex_coords;

layout(location = 0) out vec2 tex_coords_v;

void main() {
    tex_coords_v = tex_coords;
    gl_Position = pc.projection * vec4(position, 1.0);
}