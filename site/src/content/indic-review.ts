// REVIEW-ONLY Hindi clips. Untracked (.git/info/exclude) and the mp3s are gitignored: nothing here
// ships. Source: experiments/S6-indic-mint/out (Indic-Mio engine, research-only licence; README:159,
// ADR-009, CONTENT.md §6.4 and §6.7). The page must say so beside every one of these players and
// must degrade to text when a file is absent (the deployed build will not have them).
import type { CharacterId } from './alaap';

const base = import.meta.env.BASE_URL.replace(/\/$/, '');
const audio = (file: string) => `${base}/audio-indic/${file}`;

/** The three Hindi lines every S6 voice rendered. Exact text from experiments/S6-indic-mint/run_indic_mint.py. */
export const hindiLines = [
  'नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।',
  'मुझे यह फिल्म बहुत पसंद आई!',
  'हम कल सुबह जल्दी निकलेंगे।',
] as const;

/** Our plain translations; not project claims. */
export const hindiLinesEnglish = [
  'Hello, how are you? The weather is very nice today.',
  'I really liked this film!',
  'We will leave early tomorrow morning.',
] as const;

export interface IndicVoice {
  id: 'guru' | 'student' | 'narrator' | 'elder';
  /** The English demo character minted from the SAME description, or null. */
  sameDescriptionAs: CharacterId | null;
  description: string;
}

/** The S6 cast. guru/student/narrator were minted from the same three descriptions as the English demo. */
export const indicVoices: IndicVoice[] = [
  { id: 'guru', sameDescriptionAs: 'mentor', description: 'a very deep voice, very rough and gravelly, speaking very slowly' },
  { id: 'student', sameDescriptionAs: 'scout', description: 'a high voice, crisp-toned, speaking quickly and highly animated' },
  { id: 'narrator', sameDescriptionAs: 'innkeeper', description: 'a mid-range voice, warm-toned, at a steady pace' },
  { id: 'elder', sameDescriptionAs: null, description: 'a low voice, slightly hoarse, with restrained intonation' },
];

export interface IndicClip {
  id: string;
  voice: IndicVoice['id'];
  lineIndex: 0 | 1 | 2;
  text: string;
  english: string;
  seconds: number;
  /** True only for guru line 0: a fluent Hindi speaker confirmed it says the target sentence (S6 RESULTS.md:133). */
  nativeSpeakerChecked: boolean;
  src: string;
}

const secs = [4.1, 1.9, 2.1] as const;
export const indicClips: IndicClip[] = indicVoices.flatMap((v) =>
  ([0, 1, 2] as const).map((i) => ({
    id: `${v.id}_hi_line${i}`,
    voice: v.id,
    lineIndex: i,
    text: hindiLines[i],
    english: hindiLinesEnglish[i],
    seconds: secs[i],
    nativeSpeakerChecked: v.id === 'guru' && i === 0,
    src: audio(`${v.id}_hi_line${i}.mp3`),
  })),
);

/** The one sentence that must sit beside every Hindi player. */
export const indicCaveat =
  'Research render from the Indian-language engine, shown here for review only. It does not ship: the engine’s training chain includes non-commercial data.';
