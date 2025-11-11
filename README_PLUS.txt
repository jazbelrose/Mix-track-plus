Auto-DJ Mix ++

What’s new
- Smart entry points: aligns incoming tracks so a strong onset peak (drop/chorus) lands right after the crossfade.
- Target runtime: chooses a subset of your playlist to fit a time (e.g., 50 min).
- Multiple variants: generates N different mixes from the same pool while staying harmonically smooth.

Quick start
1) Put all tracks in ./input
2) venv + install deps:
   pip install -r requirements.txt
3) Run:
   python mix_tracks_plus.py --input ./input --outdir ./out      --target_bpm 172 --bars 16 --target_minutes 50      --variants 3 --harmonic_order --order_start auto      --cue_csv --use_rubberband

Outputs
- ./out/mix_v001.wav, mix_v002.wav, ... and matching *_cues.csv

Tips
- DnB: bars=16–32 for long blends. House: bars=8–16.
- If transitions hit too early/late, change --bars or edit choose_entry_offset() to move the peak ± bars.
- Rubber Band CLI improves time-stretch quality for large BPM shifts.
