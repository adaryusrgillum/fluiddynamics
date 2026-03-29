# Fluid Dynamics Simulation

An interactive WebGL-based fluid dynamics simulation inspired by modern web experiences like KIKK Festival, Wed'ze Goggles, and other creative portfolios.

## Features

- 🌊 Real-time fluid simulation using WebGL
- 🎨 Colorful, dynamic dye injection
- 🖱️ Interactive mouse and touch controls
- 📱 Fully responsive and mobile-friendly
- ⚡ GPU-accelerated for smooth performance
- 🎯 Based on Navier-Stokes equations

## Demo

Move your mouse or touch the screen to interact with the fluid simulation. The fluid responds to your movements with beautiful, flowing colors.

## Usage

Simply open `index.html` in a modern web browser that supports WebGL (Chrome, Firefox, Safari, Edge).

### Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/adaryusrgillum/fluiddynamics.git
   cd fluiddynamics
   ```

2. Open `index.html` in your browser:
   - Double-click the file, or
   - Use a local server (recommended):
     ```bash
     python -m http.server 8000
     # or
     npx serve
     ```
   - Navigate to `http://localhost:8000`

## Technical Details

### Implementation

The simulation uses:
- **WebGL shaders** for GPU-accelerated computation
- **Advection** to move dye and velocity through the fluid
- **Vorticity confinement** to preserve turbulent details
- **Pressure projection** to maintain incompressibility
- **Multiple render targets** for efficient computation

### Configuration

You can customize the simulation by modifying the `config` object in `fluid.js`:

```javascript
config: {
    simResolution: 128,          // Velocity grid resolution
    dyeResolution: 512,          // Color grid resolution
    densityDissipation: 0.98,    // How quickly color fades
    velocityDissipation: 0.99,   // How quickly velocity dampens
    pressure: 0.8,               // Pressure strength
    pressureIterations: 20,      // Pressure solver iterations
    curl: 30,                    // Vorticity strength
    splatRadius: 0.005,          // Interaction radius
    maxDeltaTime: 0.016,         // Cap timestep at 60 FPS for stability
}
```

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge
- Any browser with WebGL support

## Inspiration

This project draws inspiration from:
- [KIKK Festival](https://www.kikk.be/)
- [Wed'ze Goggles](https://www.wedzegoggles.com)
- [Julie Bonnemoy Portfolio](https://juliebonnemoy.fr/)
- [Studio Gusto](https://www.studiogusto.com/)
- [Ibiza Music Artist](https://ibizamusicartist.com)
- [Booreiland](https://booreiland.nl)
- [Republic](https://www.republic.co.uk)
- [Les Animals](https://lesanimals.tv/)

## License

MIT