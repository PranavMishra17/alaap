// Shared facts every variant imports. Words and numbers here are sourced in ../../CONTENT.md.
// Do not add claims to this file; add them to CONTENT.md with a source first.

const base = import.meta.env.BASE_URL.replace(/\/$/, '');

export const name = { latin: 'Alaap', devanagari: 'आलाप' } as const;

export const meaning =
  'the opening improvisation in Hindustani classical music, where a voice explores its full range before the composition begins';

export const links = {
  repo: 'https://github.com/PranavMishra17/alaap',
  issues: 'https://github.com/PranavMishra17/alaap/issues',
} as const;

export const languages = ['English', 'Hindi', 'Bengali', 'Tamil'] as const;

export type CharacterId = 'mentor' | 'scout' | 'innkeeper';

export interface Character {
  id: CharacterId;
  /** The full text description the voice was minted from. Nothing else was given. */
  description: string;
}

export const characters: Character[] = [
  { id: 'mentor', description: 'a very deep voice, very rough and gravelly, speaking very slowly' },
  { id: 'scout', description: 'a high voice, crisp-toned, speaking quickly and highly animated' },
  { id: 'innkeeper', description: 'a mid-range voice, warm-toned, at a steady pace' },
];

export interface Clip {
  id: string;
  character: CharacterId;
  kind: 'seed' | 'line';
  /** Exact words spoken in the clip. Always show these next to a player. */
  text: string;
  /** Seconds, rounded to 0.1. */
  seconds: number;
  /** Position in the five-line scene, or null for a seed clip. */
  lineIndex: number | null;
  /** True for the three lines rendered with the experimental, unvalidated emotion setting. */
  experimentalEmotionSetting: boolean;
  src: string;
}

const audio = (file: string) => `${base}/audio/${file}`;

export const clips: Clip[] = [
  { id: 'mentor_seed', character: 'mentor', kind: 'seed', text: 'This is how I sound when I speak.', seconds: 2.2, lineIndex: null, experimentalEmotionSetting: false, src: audio('mentor_seed.mp3') },
  { id: 'scout_seed', character: 'scout', kind: 'seed', text: 'This is how I sound when I speak.', seconds: 2.9, lineIndex: null, experimentalEmotionSetting: false, src: audio('scout_seed.mp3') },
  { id: 'innkeeper_seed', character: 'innkeeper', kind: 'seed', text: 'This is how I sound when I speak.', seconds: 1.9, lineIndex: null, experimentalEmotionSetting: false, src: audio('innkeeper_seed.mp3') },
  { id: 'mentor_line0', character: 'mentor', kind: 'line', text: 'The mountains remember every footstep, even the ones you regret.', seconds: 4.3, lineIndex: 0, experimentalEmotionSetting: false, src: audio('mentor_line0.mp3') },
  { id: 'scout_line1', character: 'scout', kind: 'line', text: "There's smoke on the ridge. Two fires, maybe three.", seconds: 3.9, lineIndex: 1, experimentalEmotionSetting: true, src: audio('scout_line1.mp3') },
  { id: 'innkeeper_line2', character: 'innkeeper', kind: 'line', text: 'Sit down, both of you. Nobody rides out on an empty stomach.', seconds: 3.5, lineIndex: 2, experimentalEmotionSetting: false, src: audio('innkeeper_line2.mp3') },
  { id: 'mentor_line3', character: 'mentor', kind: 'line', text: 'Then we leave at first light, and we do not look back.', seconds: 4.6, lineIndex: 3, experimentalEmotionSetting: true, src: audio('mentor_line3.mp3') },
  { id: 'scout_line4', character: 'scout', kind: 'line', text: 'I told you exactly what would happen!', seconds: 2.8, lineIndex: 4, experimentalEmotionSetting: true, src: audio('scout_line4.mp3') },
];

/** The three caveats that must sit near any player. Sourced in CONTENT.md §5.9, §6.7, §6.8. */
export const clipCaveats = {
  watermark: 'Every clip carries a watermark and is logged as AI-generated speech.',
  englishOnly:
    'These clips are English, from the end-to-end demo. Hindi renders exist and a native speaker has checked one, but the Indian-language engine is research-only under its licence, so those clips do not ship, not even here.',
  emotionSetting:
    'Three of the five lines were rendered with an experimental emotion setting whose effect has not been validated on this engine. Listen to them as voice samples, not performances.',
} as const;

export const scene = clips.filter((c) => c.kind === 'line').sort((a, b) => (a.lineIndex ?? 0) - (b.lineIndex ?? 0));
export const seeds = clips.filter((c) => c.kind === 'seed');
export const byCharacter = (id: CharacterId) => clips.filter((c) => c.character === id);
