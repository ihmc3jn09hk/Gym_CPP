//========================================================================
// Heightmap example program using OpenGL 3 core profile
// Copyright (c) 2010 Olivier Delannoy
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would
//    be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not
//    be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
//    distribution.
//
//========================================================================

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stddef.h>
#include <algorithm>
#include <chrono>
#include <thread>

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "gym_gl.h"
#include "gym_torch.h"

#define NUM_ITER_AT_A_TIME (1)

using namespace std::chrono_literals;

/**********************************************************************
 * Default shader programs
 *********************************************************************/

static const char* vertex_shader_text =
"#version 330\n"
"uniform mat4 project;\n"
"uniform mat4 view;\n"
"uniform mat4 model;\n"
"in vec3 v3_pos;\n"
"out float depth;\n"
//"in float x;\n"
//"in float y;\n"
//"in float z;\n"
"\n"
"void main()\n"
"{\n"
"   gl_Position = project * view * model * vec4(v3_pos, 1.0);\n"
"   depth = gl_Position.z / gl_Position.w;\n"
"   depth = 1.0 - (depth*0.5 + 0.5);\n"
"}\n";

static const char* fragment_shader_text =
"#version 330\n"
"uniform vec4 v4_color;\n"
"in float depth;\n"
"out vec4 color;\n"
"\n"
"void main()\n"
"{\n"
"    color = vec4(v4_color.rgb, depth); \n"
"}\n";

/**********************************************************************
 * Values for shader uniforms
 *********************************************************************/



/* Projection matrix */
static GLfloat projection_matrix[16] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
};

/* View matrix */
static GLfloat view_matrix[16] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
};

/* Model matrix */
static GLfloat model_matrix[16] = {
    1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 1.0f
};

/**********************************************************************
 * Heightmap vertex and index data
 *********************************************************************/
constexpr GLfloat half_width = 0.2f;
constexpr GLfloat pole_height = 2.0f;
static GLfloat pole_vertices[3 * 8] = {
	-half_width, 0.f, half_width,
     half_width, 0.f, half_width,
     half_width, pole_height, half_width,
    -half_width, pole_height, half_width,
    -half_width, 0.f, -half_width,
     half_width, 0.f, -half_width,
     half_width, pole_height, -half_width,
    -half_width, pole_height, -half_width
};
/*
static GLuint  pole_indices[4 * 6] = {
	    0, 1, 2, 3,     //Front
		5, 4, 7, 6,     //Back
		1, 5, 6, 2,     //Right
		4, 0, 3, 7,     //Left
		3, 2, 6, 7,     //Top
		1, 0, 4, 5      //Bottom
};*/
static GLuint  pole_indices[3 * 12] = {
	    1, 2, 0, 
		0, 2, 3,		//Front
		3, 2, 6,
		3, 6, 7,    	//Top
		7, 6, 4,
		4, 6, 5,		//Back
		4, 5, 0,
		0, 5, 1,   		//Bottom
		1, 5, 2,
		2, 5, 6,		//Right
		0, 3, 4, 
		4, 3, 7	        //Left
};

/* Store uniform location for the shaders
 * Those values are setup as part of the process of creating
 * the shader program. They should not be used before creating
 * the program.
 */
static GLuint pole_ibo;
static GLuint pole;
static GLuint pole_vbo;

static GLFWwindow* window;
static int iter;
static double dt;
static double last_update_time;
static int frame;
static float f;
static GLint uloc_project;
static GLint uloc_view;
static GLint uloc_model;
static GLint uloc_color;

static GLuint shader_program;
static GLfloat near = 1.0f;
constexpr double PI = 3.14159265358979323846;   // pi


/**********************************************************************
 * OpenGL helper functions
 *********************************************************************/

/* Creates a shader object of the specified type using the specified text
 */
static GLuint make_shader(GLenum type, const char* text)
{
    GLuint shader;
    GLint shader_ok;
    GLsizei log_length;
    char info_log[8192];

    shader = glCreateShader(type);
    if (shader != 0)
    {
        glShaderSource(shader, 1, (const GLchar**)&text, NULL);
        glCompileShader(shader);
        glGetShaderiv(shader, GL_COMPILE_STATUS, &shader_ok);
        if (shader_ok != GL_TRUE)
        {
            fprintf(stderr, "ERROR: Failed to compile %s shader\n", (type == GL_FRAGMENT_SHADER) ? "fragment" : "vertex" );
            glGetShaderInfoLog(shader, 8192, &log_length,info_log);
            fprintf(stderr, "ERROR: \n%s\n\n", info_log);
            glDeleteShader(shader);
            shader = 0;
        }
    }
    return shader;
}

/* Creates a program object using the specified vertex and fragment text
 */
static GLuint make_shader_program(const char* vs_text, const char* fs_text)
{
    GLuint program = 0u;
    GLint program_ok;
    GLuint vertex_shader = 0u;
    GLuint fragment_shader = 0u;
    GLsizei log_length;
    char info_log[8192];

    vertex_shader = make_shader(GL_VERTEX_SHADER, vs_text);
    if (vertex_shader != 0u)
    {
        fragment_shader = make_shader(GL_FRAGMENT_SHADER, fs_text);
        if (fragment_shader != 0u)
        {
            /* make the program that connect the two shader and link it */
            program = glCreateProgram();
            if (program != 0u)
            {
                /* attach both shader and link */
                glAttachShader(program, vertex_shader);
                glAttachShader(program, fragment_shader);
                glLinkProgram(program);
                glGetProgramiv(program, GL_LINK_STATUS, &program_ok);

                if (program_ok != GL_TRUE)
                {
                    fprintf(stderr, "ERROR, failed to link shader program\n");
                    glGetProgramInfoLog(program, 8192, &log_length, info_log);
                    fprintf(stderr, "ERROR: \n%s\n\n", info_log);
                    glDeleteProgram(program);
                    glDeleteShader(fragment_shader);
                    glDeleteShader(vertex_shader);
                    program = 0u;
                }
            }
        }
        else
        {
            fprintf(stderr, "ERROR: Unable to load fragment shader\n");
            glDeleteShader(vertex_shader);
        }
    }
    else
    {
        fprintf(stderr, "ERROR: Unable to load vertex shader\n");
    }
    return program;
}

/**********************************************************************
 * Geometry creation functions
 *********************************************************************/



/**********************************************************************
 * OpenGL helper functions
 *********************************************************************/

/* Create VBO, IBO and VAO objects for the heightmap geometry and bind them to
 * the specified program object
 */
static void gen_buffer_objects(GLuint program)
{
    GLuint attrloc;

    glGenVertexArrays(1, &pole);
    glBindVertexArray(pole);
	glGenBuffers(1, &pole_ibo);
    /* Prepare the data for drawing through a buffer inidices */
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pole_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)* 3 * 12, &pole_indices[0], GL_STATIC_DRAW);

    /* Prepare the attributes for rendering */
	glGenBuffers(1, &pole_vbo);
    attrloc = glGetAttribLocation(program, "v3_pos");
    glBindBuffer(GL_ARRAY_BUFFER, pole_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 3 * 8, &pole_vertices[0], GL_STATIC_DRAW);
    glEnableVertexAttribArray(attrloc);
    glVertexAttribPointer(attrloc, 3, GL_FLOAT, GL_FALSE, 0, 0);
}

/**********************************************************************
 * GLFW callback functions
 *********************************************************************/

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    switch(key)
    {
        case GLFW_KEY_ESCAPE:
            /* Exit program on Escape */
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
    }
}

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}


Gym_Renderer_CartPoleContinuous::Gym_Renderer_CartPoleContinuous(const int res_x, const int res_y)
	:m_iWidth(res_x),
	 m_iHeight(res_y)
{
    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
	
	window = glfwCreateWindow(m_iWidth, m_iHeight, "Gym_CPP", NULL, NULL);
    if (! window )
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    /* Register events callback */
    glfwSetKeyCallback(window, key_callback);

    glfwMakeContextCurrent(window);
    gladLoadGL(glfwGetProcAddress);

    /* Prepare opengl resources for rendering */
    shader_program = make_shader_program(vertex_shader_text, fragment_shader_text);

    if (shader_program == 0u)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
	
	glUseProgram(shader_program);
    uloc_project   = glGetUniformLocation(shader_program, "project");
    uloc_view = glGetUniformLocation(shader_program, "view");
	uloc_model = glGetUniformLocation(shader_program, "model");
	uloc_color = glGetUniformLocation(shader_program, "v4_color");
	
	/* Frustum configuration */
	// GLfloat view_angle = 45.0f;
	// GLfloat aspect_ratio = float(m_iWidth)/m_iHeight;
	// GLfloat z_near = 1.0f;
	// GLfloat z_far = z_near + pole_height + 0.1f;

	// printf("Aspect : %f", aspect_ratio);
    /* Compute the projection matrix */
    // f = 1.0f / tanf(view_angle / 2.0f);
    // projection_matrix[0]  = f / aspect_ratio;
    // projection_matrix[5]  = f;
    // projection_matrix[10] = (z_far + z_near)/ (z_near - z_far);
    // projection_matrix[11] = -1.0f;
    // projection_matrix[14] = 2.0f * (z_far * z_near) / (z_near - z_far);
	float margin = 0.5f;
	GLfloat left = -1.6f;
	GLfloat right = 1.6f;
	GLfloat top =  1.6f;
	GLfloat bottom = -1.6f;
	GLfloat z_far = pole_height + 2*margin + near;
	GLfloat z_near = std::max(0.0f, near - margin);
	
    projection_matrix[0]  = 2.0f / (right - left);
    projection_matrix[5]  = 2.0f / (top - bottom);
    projection_matrix[10] = -2.0f / (z_far - z_near);
    projection_matrix[12] = -(right + left)/(right - left);
	projection_matrix[13] = -(top + bottom)/(top - bottom);
    projection_matrix[14] = -(z_far + z_near)/(z_far - z_near);
	
    glUniformMatrix4fv(uloc_project, 1, GL_FALSE, projection_matrix);

    /* Set the camera position */
	auto view = view_matrix;
	view[4]  = view[5] = view[7] = 0.0f;
	view[6] = 1.0f;
    view[8]  = view[10] = view[11] = 0.0f;
	view[9] = -1.0f;
	
    view_matrix[12]  = 0.0f;
    view_matrix[13]  = 0.0f;
    view_matrix[14]  = -z_far;
    glUniformMatrix4fv(uloc_view, 1, GL_FALSE, view_matrix);
	
	glUniformMatrix4fv(uloc_model, 1, GL_FALSE, model_matrix);
	
	glUniform4f(uloc_color, 0.4f, 0.5f, 1.0f, 1.0f);
}

Gym_Renderer_CartPoleContinuous::~Gym_Renderer_CartPoleContinuous()
{
    glfwDestroyWindow(window);
	glfwTerminate();
}

std::pair<int,int> Gym_Renderer_CartPoleContinuous::render_state(std::vector<double> pos,
											 std::vector<double> ang,
											 std::vector<unsigned int>& data)
{
	if ( pos.size() < 1 || ang.size() < 1 ) {
		return std::pair<int,int>{0, 0};
	}

	glViewport(0, 0, m_iWidth, m_iHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
	glUseProgram(shader_program);

	//Update the model matrix
	float alpha = 0.0f;	//Rot-X
	float beta = 0.0f;	//Rot-Y
	float gamma = 0.0f;	//Rot-Z
	
	gamma = -ang[0];
	if ( 1 < ang.size()) {
		alpha = -ang[1];
	}
	float cosA = cos(alpha), sinA = sin(alpha);
	float cosB = cos(beta), sinB = sin(beta);
	float cosG = cos(gamma), sinG = sin(gamma);

	model_matrix[0] = cosB*cosG;
	model_matrix[1] = cosB*sinG;
	model_matrix[2] = -sinB;
	
	model_matrix[4] = sinA*sinB*cosG - cosA*sinG;
	model_matrix[5] = sinA*sinB*sinG + cosA*cosG;
	model_matrix[6] = sinA*cosB;
	
	model_matrix[8] = cosA*sinB*cosG + sinA*sinG;
	model_matrix[9] = cosA*sinB*sinG - sinA*cosG;
	model_matrix[10] = cosA*cosB;
	
	glUniformMatrix4fv(uloc_model, 1, GL_FALSE, model_matrix);
	
	glDrawElements(GL_TRIANGLES, 3*12, GL_UNSIGNED_INT, 0);
	 
	//glUseProgram(0);
	//glDisable(GL_DEPTH_TEST);
	data.resize(m_iWidth * m_iHeight);
    glReadPixels(0, 0, m_iWidth, m_iHeight, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
	
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
	
	glfwSwapBuffers(window);
	glfwPollEvents();
	
	return {m_iWidth, m_iHeight};
}


/**
The following is an example how to setup the renderer for Vision-based Gym.
*/
int main(int argc, char** argv)
{
	Gym_Renderer_CartPoleContinuous renderer(128, 128);
    gen_buffer_objects(shader_program);
	
	CartPole_ContinousVision gym;
	
	std::function<std::pair<int,int> (std::vector<double>,
                                                std::vector<double>,
                                                std::vector<unsigned int>&)> cb = 
	[&renderer](std::vector<double> pos, std::vector<double> ang, std::vector<unsigned int>& data)
	{
		return renderer.render_state(pos, ang, data);
	};
	
	gym.setRender_Callback(&cb);
	
    while (!glfwWindowShouldClose(window))
    {
		auto state = gym.reset();
		auto frame = 0;
		while (++frame) {
			auto action = gym.sample_action();

			auto &&rc = gym.step(action);
			auto next_state = std::get<0>(rc);
			auto reward = std::get<1>(rc);
			auto done = std::get<2>(rc);
			
			if ( done.item().toInt()){
				std::cout << "Done in : " << frame << std::endl;
				break;
			}
			/* display and process events through callbacks */

			std::this_thread::sleep_for(32ms);
		}
		//update_view_camera();
    }

    glfwTerminate();
    exit(EXIT_SUCCESS);
}

