// AVA — Tweaks panel (uses the starter components from tweaks-panel.jsx)

function AvaTweaksApp() {
  const [t, setTweak] = useTweaks(window.TWEAK_DEFAULTS || {
    mode: 'light', type: 'plex', density: 'comfortable', hero: 'ascii'
  });

  // sync to root attrs whenever values change
  React.useEffect(() => {
    if (window.AVA) {
      window.AVA.setMode(t.mode === 'dark' ? 'dark' : 'light');
      window.AVA.setType(t.type);
      window.AVA.setDensity(t.density);
      window.AVA.setHero(t.hero);
    }
  }, [t.mode, t.type, t.density, t.hero]);

  return (
    <TweaksPanel title="Tweaks">
      <TweakSection label="Background" />
      <TweakRadio
        label="Mode"
        value={t.mode}
        options={[
          { value: 'light', label: 'Paper' },
          { value: 'dark',  label: 'Phosphor' },
        ]}
        onChange={(v) => setTweak('mode', v)}
      />

      <TweakSection label="Typography" />
      <TweakSelect
        label="Pairing"
        value={t.type}
        options={[
          { value: 'plex',     label: 'IBM Plex Mono + Sans' },
          { value: 'jetinter', label: 'JetBrains Mono + Inter' },
          { value: 'berksohne',label: 'Space Mono + Grotesk' },
        ]}
        onChange={(v) => setTweak('type', v)}
      />

      <TweakSection label="Layout" />
      <TweakRadio
        label="Density"
        value={t.density}
        options={[
          { value: 'comfortable', label: 'Comfortable' },
          { value: 'compact',     label: 'Compact arxiv' },
        ]}
        onChange={(v) => setTweak('density', v)}
      />

      <TweakSection label="Hero" />
      <TweakSelect
        label="Treatment"
        value={t.hero}
        options={[
          { value: 'ascii',  label: 'ASCII frame' },
          { value: 'typing', label: 'Typing reveal' },
          { value: 'static', label: 'Static (no frame)' },
        ]}
        onChange={(v) => setTweak('hero', v)}
      />
    </TweaksPanel>
  );
}

// expose defaults to the hook
window.TWEAK_DEFAULTS = TWEAK_DEFAULTS;

const __avaTweaksRoot = document.createElement('div');
document.body.appendChild(__avaTweaksRoot);
ReactDOM.createRoot(__avaTweaksRoot).render(<AvaTweaksApp />);
