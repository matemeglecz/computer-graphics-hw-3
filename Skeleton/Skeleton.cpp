//=============================================================================================
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Meglecz Mate	
// Neptun : A7RBKU
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================


//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, tractricoid, torus, mobius, klein-bottle, boy, dini
// Camera: perspective
// Light: point or directional sources
//=============================================================================================
#include "framework.h"

//---------------------------
template<class T> struct Dnum { // Dual numbers for automatic derivation
//---------------------------
	float f; // function value
	T d;  // derivatives
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 100;

//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extrinsic
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 0.001; bp = 100;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
													u.y, v.y, w.y, 0,
													u.z, v.z, w.z, 0,
													0, 0, 0, 1);
	}

	virtual mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
					0, 1 / tan(fov / 2), 0, 0,
					0, 0, -(fp + bp) / (bp - fp), -1,
					0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};

struct OrtographicCamera : Camera{ 

public:
	mat4 P() { // projection matrix
		return mat4(0.5f , 0, 0, 0,
					0, 0.5f, 0, 0,
					0, 0, -2.0f / (bp - fp), 0,//-1.0f*(bp+fp)/(bp-fp)
					0, 0, 0, 1);
	}
};



//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;
	Material() {};
};

vec4 quaternionMultiply(vec4 q1, vec4 q2) {
	vec4 q;
	q.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
	q.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
	q.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
	q.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
	return q;
}

float length(const vec4& v) { return sqrtf(dot(v, v)); }

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 originalPos;
	vec4 wLightPos; // homogeneous coordinates, can be at ideal point
	vec4 rotationAxis;
public: 
	void Animate(float tstart, float tend) {
		float dt = tend;
		
			vec4 q = vec4(cosf(dt / 4.0f),
				sinf(dt / 4.0f) * cosf(dt) / 2.0f ,
				sinf(dt / 4.0f) * sinf(dt) / 2.0f ,
				sinf(dt / 4.0f) * sqrtf(3.0f / 4.0f));

			vec4 qinv = vec4(-1 * cosf(dt / 4.0f),
				-1 * sinf(dt / 4.0f) * cosf(dt) / 2.0f,
				-1 * sinf(dt / 4.0f) * sinf(dt) / 2.0f,
				sinf(dt / 4.0f)* sqrtf(3.0f / 4.0f));
			

		q = q/length(q);
		qinv = qinv / length(qinv);
		wLightPos = quaternionMultiply(q, originalPos - rotationAxis);

		wLightPos = quaternionMultiply(wLightPos, qinv);

		wLightPos = wLightPos + rotationAxis;
		
	}
};

//---------------------------
class CheckerBoardTexture : public Texture {
	//---------------------------
public:
	CheckerBoardTexture(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		create(width, height, image, GL_NEAREST);
	}
};

class SimpleTexture : public Texture {
public:
	SimpleTexture() : Texture() {
		std::vector<vec4> image(1);
		
		float r = (float)std::rand() / RAND_MAX ;
		float g = (float)std::rand() / RAND_MAX ;
		float b = (float)std::rand() / RAND_MAX ;

		image[0] = vec4(r, g, b, 1);
		create(1, 1, image, GL_NEAREST);
	}
};

//---------------------------
struct RenderState {
	//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light*> lights;
	Texture* texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
	//---------------------------
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};


//---------------------------
class PhongShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				float d = length(lights[i].wLightPos);
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le / pow(d, 2);
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(*state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};


//---------------------------
class Geometry {
	//---------------------------
protected:
	unsigned int vao, vbo;        // vertex array object
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
	struct VertexData {
		vec3 position, normal;
		vec2 texcoord;
	};

	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

//---------------------------
class Sphere : public ParamSurface {
	//---------------------------
public:
	Sphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
	}
};

class Hole {
public:
	float weight;
	vec2 pos;
	Hole(float w, vec2 p) : weight(w), pos(p) {
	}
};

std::vector<Hole*> holes;


class RubberPlain : public ParamSurface {
public:
	

	RubberPlain() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2 - 1;
		V = V * 2 - 1;
		X = U;
		Y = V;
		Z = 0;

		for (size_t i = 0; i < holes.size(); i++) {
			Z.f -= holes[i]->weight / (sqrtf(pow(X.f - holes[i]->pos.x, 2) + pow(Y.f - holes[i]->pos.y, 2)) + 4.0f * 0.05f);
		}
	}

};


//---------------------------
struct Object {
	//---------------------------
	Shader* shader;
	Material* material;
	Texture* texture;
	ParamSurface* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	vec3 velocity=vec3(0,0,0);
	Object(Shader* _shader, Material* _material, Texture* _texture, ParamSurface* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { 
		rotationAngle = 0.0f * tend; 
		translation = translation + velocity * (tend - tstart);
		if (abs(translation.x) > 2) {
			float dx = abs(translation.x) - 2;
			translation.x = translation.x * -1;
			if (translation.x < -2) translation.x += dx;
			else translation.x += dx;
			
		}
		if (abs(translation.y) > 2) {		
			float dy = abs(translation.y) - 2;
			translation.y = translation.y * -1;
			if (translation.y < -2) translation.y += dy;
			else translation.y += dy;
		}
	}
};

enum View{
	ortographic, projective
};

float epsilon = 0.0001;
//---------------------------
class Scene {
	//---------------------------
	std::vector<Light*> lights;
	Camera* projectiveCamera = new Camera(); // 3D camera
	OrtographicCamera* ortographicCamera = new OrtographicCamera();
public:
	View view = ortographic;
	std::vector<Object*> balls;
	Object* currentBall;
	Object* nextBall;
	Object* plainObject;
	
	void Build() {	
		// Materials
		Shader* phongShader = new PhongShader();
		Material* material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 10;

		Material* material1 = new Material;
		material1->kd = vec3(0.8f, 0.6f, 0.4f);
		material1->ks = vec3(0.3f, 0.3f, 0.3f);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 30;
		

		// Textures
		Texture* texture4x8 = new CheckerBoardTexture(4, 8);
		Texture* texture15x20 = new CheckerBoardTexture(15, 20);

		// Geometries
		ParamSurface* sphere = new Sphere();
		ParamSurface* plain = new RubberPlain();

		// Create objects by setting up their vertex data on the GPU
		Object* sphereObject1 = new Object(phongShader, material0, new SimpleTexture(), sphere);
		sphereObject1->translation = vec3(-1.8f, -1.8f, 0.1f);
		sphereObject1->scale = vec3(0.1f, 0.1f, 0.1f);
		balls.push_back(sphereObject1);
		nextBall = sphereObject1;

		plainObject = new Object(phongShader, material0, texture15x20, plain);
		//plainObject->translation = vec3(-9, 3, 0);
		plainObject->scale = vec3(2, 2, 1);
		//objects.push_back(plainObject);


		

		// Camera
		ortographicCamera->wEye = vec3(0, 0, 1);
		ortographicCamera->wLookat = vec3(0, 0, 0);
		ortographicCamera->wVup = vec3(0, 1, 0);

		projectiveCamera->wVup = vec3(0, 0, 1);

		// Lights
		lights.resize(2);
		lights[0] = new Light();
		lights[0]->wLightPos = vec4(-3.5f, 0, 1.0f, 1.0f);	// ideal point -> directional light source
		lights[0]->originalPos = vec4(-3.5f, 0, 1.0f, 1.0f);	// ideal point -> directional light source
		lights[0]->La = vec3(1.2f, 1.2f, 1.2f);
		lights[0]->Le = vec3(5, 5, 5);
		lights[0]->rotationAxis = vec4(3.5f, 0, 1.0f, 1.0f);
		//lights[0]->rotationAxis = normalize(lights[0]->rotationAxis);
		
		lights[1] = new Light();
		lights[1]->wLightPos = vec4(3.5f, 0, 1.0f, 1.0f);	// ideal point -> directional light source
		lights[1]->originalPos = vec4(3.5f, 0, 1.0f, 1.0f);	// ideal point -> directional light source
		lights[1]->La = vec3(1.2f, 1.2f, 1.2f);
		lights[1]->Le = vec3(5, 5, 5);
		lights[1]->rotationAxis = vec4(-3.5f, 0, 1.0f, 1.0f);

		
	}

	void Render() {
		RenderState state;
		if (view == ortographic) {
			state.wEye = ortographicCamera->wEye;
			state.V = ortographicCamera->V();
			state.P = ortographicCamera->P();
		}
		else {
			projectiveCamera->wEye = currentBall->translation + normalize(currentBall->velocity)* (0.1f+epsilon);
			projectiveCamera->wLookat = currentBall->velocity + currentBall->translation;
			state.wEye = projectiveCamera->wEye;
			state.V = projectiveCamera->V();
			state.P = projectiveCamera->P();
		}
		state.lights = lights;
		plainObject->Draw(state);
		for (Object* obj : balls) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		plainObject->Animate(tstart, tend);
		for (Object* obj : balls) obj->Animate(tstart, tend);
		for (Light* light : lights) light->Animate(tstart, tend);
	}
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { 
	if (key == ' ') {
		if (scene.view == ortographic && scene.currentBall!=nullptr) 
			scene.view = projective;
		else 
			scene.view = ortographic;
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { 
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	Material* material0 = new Material;
	material0->kd = vec3(0.6f, 0.4f, 0.2f);
	material0->ks = vec3(4, 4, 4);
	material0->ka = vec3(0.1f, 0.1f, 0.1f);
	material0->shininess = 10;
	
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		scene.nextBall->velocity = vec3((cX+1.0f)/2.0f*1.5f, (cY + 1.0f) / 2.0f * 1.5f, 0.0f);
		scene.currentBall = scene.nextBall;

		Object* sphereObject = new Object(new PhongShader(), material0, new SimpleTexture(), new Sphere());
		sphereObject->translation = vec3(-1.8f, -1.8f, 0.1f);
		sphereObject->scale = vec3(0.1f, 0.1f, 0.1f);
		scene.balls.push_back(sphereObject);
		scene.nextBall = sphereObject;
	}
	else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
		holes.push_back(new Hole(0.1, vec2(cX, cY)));
		
		scene.plainObject= new Object(new PhongShader(), material0, new CheckerBoardTexture(15, 20), new RubberPlain());
		scene.plainObject->scale = vec3(2, 2, 1);

			
	}
	
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1f; // dt is �infinitesimal�
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}