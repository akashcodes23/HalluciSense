"use client";

import React, { useRef, useMemo, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

interface Hero3DCanvasProps {
  heroState?: "detect" | "confidence" | "verify";
  reducedMotion?: boolean;
}

/* ── Nested Gyroscope Rings ────────────────────────────────────────────────── */
function GyroscopeRings({ heroState, reducedMotion }: { heroState: "detect" | "confidence" | "verify"; reducedMotion: boolean }) {
  const ring1Ref = useRef<THREE.Group>(null);
  const ring2Ref = useRef<THREE.Group>(null);
  const ring3Ref = useRef<THREE.Group>(null);

  useFrame((_, delta) => {
    if (reducedMotion) return;

    // Adjust rotation speeds based on active hero state
    let speedMult = 1.0;
    if (heroState === "detect") speedMult = 1.4;
    else if (heroState === "confidence") speedMult = 0.6;
    else if (heroState === "verify") speedMult = 0.9;

    if (ring1Ref.current) {
      ring1Ref.current.rotation.x += delta * 0.25 * speedMult;
      ring1Ref.current.rotation.y += delta * 0.15 * speedMult;
    }
    if (ring2Ref.current) {
      ring2Ref.current.rotation.y -= delta * 0.20 * speedMult;
      ring2Ref.current.rotation.z += delta * 0.18 * speedMult;
    }
    if (ring3Ref.current) {
      ring3Ref.current.rotation.z += delta * 0.22 * speedMult;
      ring3Ref.current.rotation.x -= delta * 0.12 * speedMult;
    }
  });

  const ringColor = useMemo(() => {
    if (heroState === "detect") return "#10b981"; // Emerald
    if (heroState === "confidence") return "#14b8a6"; // Teal
    return "#06b6d4"; // Cyan
  }, [heroState]);

  return (
    <group>
      {/* Outer Gyroscope Ring */}
      <group ref={ring1Ref}>
        <mesh>
          <torusGeometry args={[2.2, 0.012, 16, 100]} />
          <meshBasicMaterial color={ringColor} transparent opacity={0.35} wireframe={false} />
        </mesh>
      </group>

      {/* Middle Gyroscope Ring */}
      <group ref={ring2Ref}>
        <mesh rotation={[Math.PI / 4, 0, 0]}>
          <torusGeometry args={[1.85, 0.010, 16, 100]} />
          <meshBasicMaterial color={ringColor} transparent opacity={0.45} />
        </mesh>
      </group>

      {/* Inner Precision Ring */}
      <group ref={ring3Ref}>
        <mesh rotation={[0, Math.PI / 3, Math.PI / 6]}>
          <torusGeometry args={[1.5, 0.008, 16, 80]} />
          <meshBasicMaterial color="#ffffff" transparent opacity={0.5} />
        </mesh>
      </group>
    </group>
  );
}

/* ── Holographic Neural Cognition Core ─────────────────────────────────────── */
function NeuralCognitionCore({ heroState, reducedMotion }: { heroState: "detect" | "confidence" | "verify"; reducedMotion: boolean }) {
  const pointsRef = useRef<THREE.Points>(null);
  const linesRef = useRef<THREE.LineSegments>(null);
  const groupRef = useRef<THREE.Group>(null);

  // Generate 450 procedural nodes on a structured spherical point cloud
  const { positions, linePositions } = useMemo(() => {
    const count = 420;
    const pos = new Float32Array(count * 3);
    const radius = 1.1;

    // Fibonacci sphere distribution
    const phi = Math.PI * (3 - Math.sqrt(5)); // Golden angle
    for (let i = 0; i < count; i++) {
      const y = 1 - (i / (count - 1)) * 2; // y goes from 1 to -1
      const radiusAtY = Math.sqrt(1 - y * y);
      const theta = phi * i;

      // Add gentle radial organic perturbation
      const r = radius * (0.85 + 0.3 * Math.sin(i * 1.5));
      const x = Math.cos(theta) * radiusAtY * r;
      const z = Math.sin(theta) * radiusAtY * r;

      pos[i * 3] = x;
      pos[i * 3 + 1] = y * r;
      pos[i * 3 + 2] = z;
    }

    // Connect close neighbors with line segments (max 2 per point for performance)
    const lineCoords: number[] = [];
    for (let i = 0; i < count; i++) {
      for (let j = i + 1; j < count; j++) {
        const dx = pos[i * 3] - pos[j * 3];
        const dy = pos[i * 3 + 1] - pos[j * 3 + 1];
        const dz = pos[i * 3 + 2] - pos[j * 3 + 2];
        const distSq = dx * dx + dy * dy + dz * dz;

        if (distSq < 0.12) {
          lineCoords.push(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
          lineCoords.push(pos[j * 3], pos[j * 3 + 1], pos[j * 3 + 2]);
        }
      }
    }

    return {
      positions: pos,
      linePositions: new Float32Array(lineCoords),
    };
  }, []);

  useFrame((state, delta) => {
    if (reducedMotion) return;

    const t = state.clock.elapsedTime;

    if (groupRef.current) {
      // Rotation based on state
      groupRef.current.rotation.y += delta * 0.15;
      groupRef.current.rotation.x = Math.sin(t * 0.5) * 0.1;
    }

    if (pointsRef.current) {
      // Subtle pulsation
      const pulseRate = heroState === "detect" ? 2.5 : heroState === "confidence" ? 1.2 : 1.8;
      const scale = 1.0 + Math.sin(t * pulseRate) * 0.04;
      pointsRef.current.scale.set(scale, scale, scale);
    }
  });

  const coreColor = useMemo(() => {
    if (heroState === "detect") return new THREE.Color("#34d399"); // Emerald 400
    if (heroState === "confidence") return new THREE.Color("#2dd4bf"); // Teal 400
    return new THREE.Color("#38bdf8"); // Sky / Cyan 400
  }, [heroState]);

  return (
    <group ref={groupRef}>
      {/* Node Particles */}
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[positions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.035}
          color={coreColor}
          transparent
          opacity={0.85}
          sizeAttenuation
          blending={THREE.AdditiveBlending}
        />
      </points>

      {/* Network Connectors */}
      <lineSegments ref={linesRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[linePositions, 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color={coreColor}
          transparent
          opacity={0.22}
          blending={THREE.AdditiveBlending}
        />
      </lineSegments>

      {/* Central Concentrated Glow Core */}
      <mesh>
        <sphereGeometry args={[0.35, 16, 16]} />
        <meshBasicMaterial color={coreColor} transparent opacity={0.15} />
      </mesh>
    </group>
  );
}

/* ── Master Hero 3D Scene ──────────────────────────────────────────────────── */
function ScientificScene({ heroState, reducedMotion }: { heroState: "detect" | "confidence" | "verify"; reducedMotion: boolean }) {
  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[4, 3, 5]} intensity={1.2} color="#10b981" />
      <pointLight position={[-4, -3, -3]} intensity={0.8} color="#14b8a6" />
      <pointLight position={[0, 4, -2]} intensity={0.6} color="#06b6d4" />

      <group position={[0, 0, 0]}>
        <GyroscopeRings heroState={heroState} reducedMotion={reducedMotion} />
        <NeuralCognitionCore heroState={heroState} reducedMotion={reducedMotion} />
      </group>
    </>
  );
}

/* ── Exported Component with Lightweight Fallback ──────────────────────────── */
export default function Hero3DCanvas({
  heroState = "detect",
  reducedMotion = false,
}: Hero3DCanvasProps) {
  return (
    <div className="relative w-full h-[380px] sm:h-[440px] lg:h-[500px] flex items-center justify-center pointer-events-none select-none">
      {/* Background Volumetric Glow */}
      <div className="absolute inset-0 bg-radial from-emerald-500/10 via-teal-500/5 to-transparent blur-3xl opacity-70" />

      {/* WebGL Canvas */}
      <Canvas
        camera={{ position: [0, 0, 4.8], fov: 42 }}
        dpr={[1, 1.5]}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: "high-performance",
        }}
        className="w-full h-full"
      >
        <ScientificScene heroState={heroState} reducedMotion={reducedMotion} />
      </Canvas>
    </div>
  );
}
