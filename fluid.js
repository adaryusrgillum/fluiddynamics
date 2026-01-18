// Fluid Dynamics Simulation using WebGL
// Based on GPU-accelerated fluid simulation techniques

class FluidSimulation {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        
        if (!this.gl) {
            console.error('WebGL not supported');
            return;
        }

        this.config = {
            simResolution: 128,
            dyeResolution: 512,
            densityDissipation: 0.98,
            velocityDissipation: 0.99,
            pressure: 0.8,
            pressureIterations: 20,
            curl: 30,
            splatRadius: 0.005,
            maxDeltaTime: 0.016, // Cap timestep at 60 FPS for stability
            colorPalette: [
                [1.0, 0.0, 0.5],
                [0.0, 0.5, 1.0],
                [0.5, 1.0, 0.0],
                [1.0, 0.5, 0.0],
                [0.5, 0.0, 1.0],
            ]
        };

        this.pointers = [];
        this.lastTime = performance.now();
        
        this.resizeCanvas();
        this.initPrograms();
        this.initFramebuffers();
        this.setupEventListeners();
        this.animate();
    }

    resizeCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.aspectRatio = this.canvas.width / this.canvas.height;
    }

    compileShader(type, source) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);

        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error('Shader compilation error:', this.gl.getShaderInfoLog(shader));
            this.gl.deleteShader(shader);
            return null;
        }

        return shader;
    }

    createProgram(vertexShader, fragmentShader) {
        const program = this.gl.createProgram();
        this.gl.attachShader(program, vertexShader);
        this.gl.attachShader(program, fragmentShader);
        
        // Bind attribute location before linking
        this.gl.bindAttribLocation(program, 0, 'aPosition');
        
        this.gl.linkProgram(program);

        if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
            console.error('Program linking error:', this.gl.getProgramInfoLog(program));
            return null;
        }

        return program;
    }

    initPrograms() {
        const baseVertexShader = `
            precision highp float;
            attribute vec2 aPosition;
            varying vec2 vUv;
            varying vec2 vL;
            varying vec2 vR;
            varying vec2 vT;
            varying vec2 vB;
            uniform vec2 texelSize;

            void main () {
                vUv = aPosition * 0.5 + 0.5;
                vL = vUv - vec2(texelSize.x, 0.0);
                vR = vUv + vec2(texelSize.x, 0.0);
                vT = vUv + vec2(0.0, texelSize.y);
                vB = vUv - vec2(0.0, texelSize.y);
                gl_Position = vec4(aPosition, 0.0, 1.0);
            }
        `;

        const displayShader = `
            precision highp float;
            varying vec2 vUv;
            uniform sampler2D uTexture;

            void main () {
                vec3 color = texture2D(uTexture, vUv).rgb;
                gl_FragColor = vec4(color, 1.0);
            }
        `;

        const splatShader = `
            precision highp float;
            varying vec2 vUv;
            uniform sampler2D uTarget;
            uniform float aspectRatio;
            uniform vec3 color;
            uniform vec2 point;
            uniform float radius;

            void main () {
                vec2 p = vUv - point.xy;
                p.x *= aspectRatio;
                vec3 splat = exp(-dot(p, p) / radius) * color;
                vec3 base = texture2D(uTarget, vUv).xyz;
                gl_FragColor = vec4(base + splat, 1.0);
            }
        `;

        const advectionShader = `
            precision highp float;
            varying vec2 vUv;
            uniform sampler2D uVelocity;
            uniform sampler2D uSource;
            uniform vec2 texelSize;
            uniform float dt;
            uniform float dissipation;

            void main () {
                vec2 coord = vUv - dt * texture2D(uVelocity, vUv).rg * texelSize;
                gl_FragColor = dissipation * texture2D(uSource, coord);
            }
        `;

        const divergenceShader = `
            precision highp float;
            varying highp vec2 vUv;
            varying highp vec2 vL;
            varying highp vec2 vR;
            varying highp vec2 vT;
            varying highp vec2 vB;
            uniform sampler2D uVelocity;

            void main () {
                float L = texture2D(uVelocity, vL).r;
                float R = texture2D(uVelocity, vR).r;
                float T = texture2D(uVelocity, vT).g;
                float B = texture2D(uVelocity, vB).g;
                float div = 0.5 * (R - L + T - B);
                gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
            }
        `;

        const curlShader = `
            precision highp float;
            varying highp vec2 vUv;
            varying highp vec2 vL;
            varying highp vec2 vR;
            varying highp vec2 vT;
            varying highp vec2 vB;
            uniform sampler2D uVelocity;

            void main () {
                float L = texture2D(uVelocity, vL).g;
                float R = texture2D(uVelocity, vR).g;
                float T = texture2D(uVelocity, vT).r;
                float B = texture2D(uVelocity, vB).r;
                float vorticity = R - L - T + B;
                gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
            }
        `;

        const vorticityShader = `
            precision highp float;
            varying vec2 vUv;
            varying vec2 vL;
            varying vec2 vR;
            varying vec2 vT;
            varying vec2 vB;
            uniform sampler2D uVelocity;
            uniform sampler2D uCurl;
            uniform float curl;
            uniform float dt;

            void main () {
                float L = texture2D(uCurl, vL).r;
                float R = texture2D(uCurl, vR).r;
                float T = texture2D(uCurl, vT).r;
                float B = texture2D(uCurl, vB).r;
                float C = texture2D(uCurl, vUv).r;
                vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
                force /= length(force) + 0.0001;
                force *= curl * C;
                force.y *= -1.0;
                vec2 vel = texture2D(uVelocity, vUv).rg;
                gl_FragColor = vec4(vel + force * dt, 0.0, 1.0);
            }
        `;

        const pressureShader = `
            precision highp float;
            varying highp vec2 vUv;
            varying highp vec2 vL;
            varying highp vec2 vR;
            varying highp vec2 vT;
            varying highp vec2 vB;
            uniform sampler2D uPressure;
            uniform sampler2D uDivergence;

            void main () {
                float L = texture2D(uPressure, vL).r;
                float R = texture2D(uPressure, vR).r;
                float T = texture2D(uPressure, vT).r;
                float B = texture2D(uPressure, vB).r;
                float C = texture2D(uPressure, vUv).r;
                float divergence = texture2D(uDivergence, vUv).r;
                float pressure = (L + R + B + T - divergence) * 0.25;
                gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
            }
        `;

        const gradientSubtractShader = `
            precision highp float;
            varying highp vec2 vUv;
            varying highp vec2 vL;
            varying highp vec2 vR;
            varying highp vec2 vT;
            varying highp vec2 vB;
            uniform sampler2D uPressure;
            uniform sampler2D uVelocity;

            void main () {
                float L = texture2D(uPressure, vL).r;
                float R = texture2D(uPressure, vR).r;
                float T = texture2D(uPressure, vT).r;
                float B = texture2D(uPressure, vB).r;
                vec2 velocity = texture2D(uVelocity, vUv).rg;
                velocity.xy -= vec2(R - L, T - B);
                gl_FragColor = vec4(velocity, 0.0, 1.0);
            }
        `;

        const vertShader = this.compileShader(this.gl.VERTEX_SHADER, baseVertexShader);

        this.programs = {
            display: this.createProgram(vertShader, this.compileShader(this.gl.FRAGMENT_SHADER, displayShader)),
            splat: this.createProgram(vertShader, this.compileShader(this.gl.FRAGMENT_SHADER, splatShader)),
            advection: this.createProgram(vertShader, this.compileShader(this.gl.FRAGMENT_SHADER, advectionShader)),
            divergence: this.createProgram(vertShader, this.compileShader(this.gl.FRAGMENT_SHADER, divergenceShader)),
            curl: this.createProgram(vertShader, this.compileShader(this.gl.FRAGMENT_SHADER, curlShader)),
            vorticity: this.createProgram(vertShader, this.compileShader(this.gl.FRAGMENT_SHADER, vorticityShader)),
            pressure: this.createProgram(vertShader, this.compileShader(this.gl.FRAGMENT_SHADER, pressureShader)),
            gradientSubtract: this.createProgram(vertShader, this.compileShader(this.gl.FRAGMENT_SHADER, gradientSubtractShader)),
        };

        // Full-screen quad vertices: [bottom-left, bottom-right, top-right, top-left]
        const FULL_SCREEN_QUAD_VERTICES = new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]);
        
        // Create a quad buffer for rendering
        const quadBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, quadBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, FULL_SCREEN_QUAD_VERTICES, this.gl.STATIC_DRAW);
        
        // Store for later use
        this.quadBuffer = quadBuffer;
    }

    createFBO(w, h, type, format, internalFormat) {
        const gl = this.gl;
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

        const fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
        gl.viewport(0, 0, w, h);
        gl.clear(gl.COLOR_BUFFER_BIT);

        return {
            texture,
            fbo,
            width: w,
            height: h
        };
    }

    createDoubleFBO(w, h, type, format, internalFormat) {
        let fbo1 = this.createFBO(w, h, type, format, internalFormat);
        let fbo2 = this.createFBO(w, h, type, format, internalFormat);

        return {
            get read() { return fbo1; },
            set read(value) { fbo1 = value; },
            get write() { return fbo2; },
            set write(value) { fbo2 = value; },
            swap() {
                let temp = fbo1;
                fbo1 = fbo2;
                fbo2 = temp;
            }
        };
    }

    initFramebuffers() {
        const gl = this.gl;
        const simRes = this.config.simResolution;
        const dyeRes = this.config.dyeResolution;

        // Check for float texture support
        const ext = gl.getExtension('OES_texture_float') || 
                    gl.getExtension('OES_texture_half_float') ||
                    gl.getExtension('EXT_color_buffer_float');
        
        const texType = ext ? gl.FLOAT : gl.UNSIGNED_BYTE;
        const rgba = gl.RGBA;
        const rgba16f = gl.RGBA;

        this.density = this.createDoubleFBO(dyeRes, dyeRes, texType, rgba, rgba16f);
        this.velocity = this.createDoubleFBO(simRes, simRes, texType, rgba, rgba16f);
        this.divergence = this.createFBO(simRes, simRes, texType, rgba, rgba16f);
        this.curl = this.createFBO(simRes, simRes, texType, rgba, rgba16f);
        this.pressure = this.createDoubleFBO(simRes, simRes, texType, rgba, rgba16f);
    }

    setupEventListeners() {
        window.addEventListener('resize', () => {
            this.resizeCanvas();
        });

        this.canvas.addEventListener('mousemove', (e) => {
            const pointer = this.pointers.find(p => p.id === -1) || { id: -1, down: false, moved: false, dx: 0, dy: 0, x: 0, y: 0, color: null };
            
            if (!this.pointers.includes(pointer)) {
                this.pointers.push(pointer);
            }

            pointer.moved = pointer.down;
            pointer.dx = (e.clientX - pointer.x) * 5.0;
            pointer.dy = (e.clientY - pointer.y) * 5.0;
            pointer.x = e.clientX;
            pointer.y = e.clientY;
        });

        this.canvas.addEventListener('mousedown', (e) => {
            const pointer = this.pointers.find(p => p.id === -1) || { id: -1, down: false, moved: false, dx: 0, dy: 0, x: 0, y: 0, color: null };
            
            if (!this.pointers.includes(pointer)) {
                this.pointers.push(pointer);
            }

            pointer.down = true;
            pointer.moved = false;
            pointer.x = e.clientX;
            pointer.y = e.clientY;
            pointer.color = this.config.colorPalette[Math.floor(Math.random() * this.config.colorPalette.length)];
        });

        this.canvas.addEventListener('mouseup', () => {
            const pointer = this.pointers.find(p => p.id === -1);
            if (pointer) pointer.down = false;
        });

        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touches = e.targetTouches;
            for (let i = 0; i < touches.length; i++) {
                const touch = touches[i];
                let pointer = this.pointers.find(p => p.id === touch.identifier);
                
                if (!pointer) {
                    pointer = { id: touch.identifier, down: false, moved: false, dx: 0, dy: 0, x: 0, y: 0, color: null };
                    this.pointers.push(pointer);
                }

                pointer.down = true;
                pointer.moved = false;
                pointer.x = touch.clientX;
                pointer.y = touch.clientY;
                pointer.color = this.config.colorPalette[Math.floor(Math.random() * this.config.colorPalette.length)];
            }
        }, { passive: false });

        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touches = e.targetTouches;
            for (let i = 0; i < touches.length; i++) {
                const touch = touches[i];
                const pointer = this.pointers.find(p => p.id === touch.identifier);
                
                if (pointer) {
                    pointer.moved = pointer.down;
                    pointer.dx = (touch.clientX - pointer.x) * 8.0;
                    pointer.dy = (touch.clientY - pointer.y) * 8.0;
                    pointer.x = touch.clientX;
                    pointer.y = touch.clientY;
                }
            }
        }, { passive: false });

        this.canvas.addEventListener('touchend', (e) => {
            const touches = e.changedTouches;
            for (let i = 0; i < touches.length; i++) {
                const touch = touches[i];
                const pointer = this.pointers.find(p => p.id === touch.identifier);
                if (pointer) pointer.down = false;
            }
        });
    }

    splat(x, y, dx, dy, color) {
        const gl = this.gl;
        gl.viewport(0, 0, this.velocity.read.width, this.velocity.read.height);
        
        gl.useProgram(this.programs.splat);
        gl.uniform1i(gl.getUniformLocation(this.programs.splat, 'uTarget'), 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
        gl.uniform1f(gl.getUniformLocation(this.programs.splat, 'aspectRatio'), this.aspectRatio);
        gl.uniform2f(gl.getUniformLocation(this.programs.splat, 'point'), x / this.canvas.width, 1.0 - y / this.canvas.height);
        gl.uniform3f(gl.getUniformLocation(this.programs.splat, 'color'), dx, -dy, 1.0);
        gl.uniform1f(gl.getUniformLocation(this.programs.splat, 'radius'), this.config.splatRadius);
        
        this.blit(this.velocity.write.fbo);
        this.velocity.swap();

        gl.viewport(0, 0, this.density.read.width, this.density.read.height);
        gl.uniform1i(gl.getUniformLocation(this.programs.splat, 'uTarget'), 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.density.read.texture);
        gl.uniform3f(gl.getUniformLocation(this.programs.splat, 'color'), color[0] * 0.3, color[1] * 0.3, color[2] * 0.3);
        
        this.blit(this.density.write.fbo);
        this.density.swap();
    }

    blit(destination) {
        const gl = this.gl;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(0);
        
        if (destination) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, destination);
        } else {
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        }
        
        gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
    }

    update() {
        const gl = this.gl;
        
        // Calculate dynamic timestep based on actual frame time
        const currentTime = performance.now();
        const dt = Math.min((currentTime - this.lastTime) / 1000, this.config.maxDeltaTime);
        this.lastTime = currentTime;

        // Apply inputs
        for (let i = 0; i < this.pointers.length; i++) {
            const pointer = this.pointers[i];
            if (pointer.moved) {
                this.splat(pointer.x, pointer.y, pointer.dx, pointer.dy, pointer.color);
                pointer.moved = false;
            }
        }

        // Curl
        gl.viewport(0, 0, this.curl.width, this.curl.height);
        gl.useProgram(this.programs.curl);
        gl.uniform2f(gl.getUniformLocation(this.programs.curl, 'texelSize'), 1.0 / this.velocity.read.width, 1.0 / this.velocity.read.height);
        gl.uniform1i(gl.getUniformLocation(this.programs.curl, 'uVelocity'), 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
        this.blit(this.curl.fbo);

        // Vorticity
        gl.viewport(0, 0, this.velocity.read.width, this.velocity.read.height);
        gl.useProgram(this.programs.vorticity);
        gl.uniform2f(gl.getUniformLocation(this.programs.vorticity, 'texelSize'), 1.0 / this.velocity.read.width, 1.0 / this.velocity.read.height);
        gl.uniform1i(gl.getUniformLocation(this.programs.vorticity, 'uVelocity'), 0);
        gl.uniform1i(gl.getUniformLocation(this.programs.vorticity, 'uCurl'), 1);
        gl.uniform1f(gl.getUniformLocation(this.programs.vorticity, 'curl'), this.config.curl);
        gl.uniform1f(gl.getUniformLocation(this.programs.vorticity, 'dt'), dt);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.curl.texture);
        this.blit(this.velocity.write.fbo);
        this.velocity.swap();

        // Divergence
        gl.viewport(0, 0, this.divergence.width, this.divergence.height);
        gl.useProgram(this.programs.divergence);
        gl.uniform2f(gl.getUniformLocation(this.programs.divergence, 'texelSize'), 1.0 / this.velocity.read.width, 1.0 / this.velocity.read.height);
        gl.uniform1i(gl.getUniformLocation(this.programs.divergence, 'uVelocity'), 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
        this.blit(this.divergence.fbo);

        // Pressure
        gl.viewport(0, 0, this.pressure.read.width, this.pressure.read.height);
        gl.useProgram(this.programs.pressure);
        gl.uniform2f(gl.getUniformLocation(this.programs.pressure, 'texelSize'), 1.0 / this.pressure.read.width, 1.0 / this.pressure.read.height);
        gl.uniform1i(gl.getUniformLocation(this.programs.pressure, 'uDivergence'), 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.divergence.texture);
        
        for (let i = 0; i < this.config.pressureIterations; i++) {
            gl.uniform1i(gl.getUniformLocation(this.programs.pressure, 'uPressure'), 1);
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.texture);
            this.blit(this.pressure.write.fbo);
            this.pressure.swap();
        }

        // Gradient subtract
        gl.viewport(0, 0, this.velocity.read.width, this.velocity.read.height);
        gl.useProgram(this.programs.gradientSubtract);
        gl.uniform2f(gl.getUniformLocation(this.programs.gradientSubtract, 'texelSize'), 1.0 / this.velocity.read.width, 1.0 / this.velocity.read.height);
        gl.uniform1i(gl.getUniformLocation(this.programs.gradientSubtract, 'uPressure'), 0);
        gl.uniform1i(gl.getUniformLocation(this.programs.gradientSubtract, 'uVelocity'), 1);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.pressure.read.texture);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
        this.blit(this.velocity.write.fbo);
        this.velocity.swap();

        // Advect velocity
        gl.viewport(0, 0, this.velocity.read.width, this.velocity.read.height);
        gl.useProgram(this.programs.advection);
        gl.uniform2f(gl.getUniformLocation(this.programs.advection, 'texelSize'), 1.0 / this.velocity.read.width, 1.0 / this.velocity.read.height);
        gl.uniform1i(gl.getUniformLocation(this.programs.advection, 'uVelocity'), 0);
        gl.uniform1i(gl.getUniformLocation(this.programs.advection, 'uSource'), 0);
        gl.uniform1f(gl.getUniformLocation(this.programs.advection, 'dt'), dt);
        gl.uniform1f(gl.getUniformLocation(this.programs.advection, 'dissipation'), this.config.velocityDissipation);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
        this.blit(this.velocity.write.fbo);
        this.velocity.swap();

        // Advect density
        gl.viewport(0, 0, this.density.read.width, this.density.read.height);
        gl.useProgram(this.programs.advection);
        gl.uniform2f(gl.getUniformLocation(this.programs.advection, 'texelSize'), 1.0 / this.density.read.width, 1.0 / this.density.read.height);
        gl.uniform1i(gl.getUniformLocation(this.programs.advection, 'uVelocity'), 0);
        gl.uniform1i(gl.getUniformLocation(this.programs.advection, 'uSource'), 1);
        gl.uniform1f(gl.getUniformLocation(this.programs.advection, 'dt'), dt);
        gl.uniform1f(gl.getUniformLocation(this.programs.advection, 'dissipation'), this.config.densityDissipation);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.velocity.read.texture);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.density.read.texture);
        this.blit(this.density.write.fbo);
        this.density.swap();
    }

    render() {
        const gl = this.gl;
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.useProgram(this.programs.display);
        gl.uniform1i(gl.getUniformLocation(this.programs.display, 'uTexture'), 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.density.read.texture);
        this.blit(null);
    }

    animate() {
        this.update();
        this.render();
        requestAnimationFrame(() => this.animate());
    }
}

// Initialize the simulation when the page loads
window.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('fluid-canvas');
    new FluidSimulation(canvas);
});
