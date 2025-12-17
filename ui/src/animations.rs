//! Animation utilities for smooth UI transitions

use std::time::Instant;

/// Animated value with easing
#[derive(Clone)]
pub struct AnimatedValue {
    start: f32,
    end: f32,
    current: f32,
    start_time: Instant,
    duration_ms: u64,
    easing: EasingType,
}

#[derive(Clone, Copy, Debug)]
pub enum EasingType {
    Linear,
    EaseOutExpo,
    EaseOutCubic,
    EaseInOutQuad,
}

impl AnimatedValue {
    pub fn new(initial: f32) -> Self {
        Self {
            start: initial,
            end: initial,
            current: initial,
            start_time: Instant::now(),
            duration_ms: 300,
            easing: EasingType::EaseOutCubic,
        }
    }
    
    pub fn animate_to(&mut self, target: f32, duration_ms: u64) {
        if (self.end - target).abs() > 0.001 {
            self.start = self.current;
            self.end = target;
            self.start_time = Instant::now();
            self.duration_ms = duration_ms;
        }
    }
    
    pub fn set_easing(&mut self, easing: EasingType) {
        self.easing = easing;
    }
    
    pub fn get(&mut self) -> f32 {
        let elapsed = self.start_time.elapsed().as_millis() as f32;
        let duration = self.duration_ms as f32;
        
        if elapsed >= duration {
            self.current = self.end;
            return self.end;
        }
        
        let t = elapsed / duration;
        let eased = match self.easing {
            EasingType::Linear => t,
            EasingType::EaseOutExpo => if t >= 1.0 { 1.0 } else { 1.0 - 2.0_f32.powf(-10.0 * t) },
            EasingType::EaseOutCubic => 1.0 - (1.0 - t).powi(3),
            EasingType::EaseInOutQuad => {
                if t < 0.5 { 2.0 * t * t } else { 1.0 - (-2.0 * t + 2.0).powi(2) / 2.0 }
            }
        };
        
        self.current = self.start + (self.end - self.start) * eased;
        self.current
    }
    
    pub fn is_animating(&self) -> bool {
        self.start_time.elapsed().as_millis() < self.duration_ms as u128
    }
}

/// Pulsing animation for "breathing" effects
pub struct PulseAnimation {
    frequency: f32,
    amplitude: f32,
    offset: f32,
    start_time: Instant,
}

impl PulseAnimation {
    pub fn new(frequency: f32, amplitude: f32) -> Self {
        Self {
            frequency,
            amplitude,
            offset: 0.5,
            start_time: Instant::now(),
        }
    }
    
    pub fn get(&self) -> f32 {
        let t = self.start_time.elapsed().as_secs_f32();
        let wave = (t * self.frequency * std::f32::consts::PI * 2.0).sin();
        self.offset + wave * self.amplitude
    }
}

/// Spring physics animation
pub struct SpringAnimation {
    position: f32,
    velocity: f32,
    target: f32,
    stiffness: f32,
    damping: f32,
    last_update: Instant,
}

impl SpringAnimation {
    pub fn new(initial: f32, stiffness: f32, damping: f32) -> Self {
        Self {
            position: initial,
            velocity: 0.0,
            target: initial,
            stiffness,
            damping,
            last_update: Instant::now(),
        }
    }
    
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }
    
    pub fn update(&mut self) -> f32 {
        let dt = self.last_update.elapsed().as_secs_f32().min(0.1);
        self.last_update = Instant::now();
        
        let displacement = self.target - self.position;
        let spring_force = displacement * self.stiffness;
        let damping_force = -self.velocity * self.damping;
        let acceleration = spring_force + damping_force;
        
        self.velocity += acceleration * dt;
        self.position += self.velocity * dt;
        
        self.position
    }
    
    pub fn is_settled(&self) -> bool {
        (self.target - self.position).abs() < 0.001 && self.velocity.abs() < 0.001
    }
}

/// Typewriter animation for text
pub struct TypewriterAnimation {
    full_text: String,
    revealed_chars: usize,
    chars_per_second: f32,
    last_update: Instant,
    accumulated_time: f32,
}

impl TypewriterAnimation {
    pub fn new(text: &str, chars_per_second: f32) -> Self {
        Self {
            full_text: text.to_string(),
            revealed_chars: 0,
            chars_per_second,
            last_update: Instant::now(),
            accumulated_time: 0.0,
        }
    }
    
    pub fn update(&mut self) -> &str {
        let dt = self.last_update.elapsed().as_secs_f32();
        self.last_update = Instant::now();
        self.accumulated_time += dt;
        
        let chars_to_reveal = (self.accumulated_time * self.chars_per_second) as usize;
        self.revealed_chars = chars_to_reveal.min(self.full_text.chars().count());
        
        &self.full_text[..self.full_text.char_indices()
            .nth(self.revealed_chars)
            .map(|(i, _)| i)
            .unwrap_or(self.full_text.len())]
    }
    
    pub fn is_complete(&self) -> bool {
        self.revealed_chars >= self.full_text.chars().count()
    }
    
    pub fn skip_to_end(&mut self) {
        self.revealed_chars = self.full_text.chars().count();
    }
}

/// Particle system for visual effects
pub struct Particle {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub life: f32,
    pub max_life: f32,
    pub size: f32,
    pub color: [f32; 4],
}

pub struct ParticleSystem {
    particles: Vec<Particle>,
    max_particles: usize,
    last_update: Instant,
}

impl ParticleSystem {
    pub fn new(max_particles: usize) -> Self {
        Self {
            particles: Vec::with_capacity(max_particles),
            max_particles,
            last_update: Instant::now(),
        }
    }
    
    pub fn emit(&mut self, x: f32, y: f32, count: usize, color: [f32; 4]) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for _ in 0..count {
            if self.particles.len() >= self.max_particles {
                break;
            }
            
            let angle = rng.gen::<f32>() * std::f32::consts::PI * 2.0;
            let speed = rng.gen::<f32>() * 50.0 + 20.0;
            
            self.particles.push(Particle {
                x,
                y,
                vx: angle.cos() * speed,
                vy: angle.sin() * speed,
                life: 1.0,
                max_life: 1.0,
                size: rng.gen::<f32>() * 3.0 + 1.0,
                color,
            });
        }
    }
    
    pub fn update(&mut self) {
        let dt = self.last_update.elapsed().as_secs_f32();
        self.last_update = Instant::now();
        
        self.particles.retain_mut(|p| {
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.life -= dt / p.max_life;
            p.life > 0.0
        });
    }
    
    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }
}
