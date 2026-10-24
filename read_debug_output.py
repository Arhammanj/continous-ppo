from pathlib import Path
p = Path('debug_reward_out.txt')
text = None
for enc in ('utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1'):
    try:
        text = p.read_text(encoding=enc)
        print(f'Decoded with {enc}, length {len(text)}')
        break
    except Exception as e:
        print(f'Failed {enc}: {e}')

if text is None:
    print('Could not decode file')
else:
    out = Path('debug_reward_out_decoded.txt')
    out.write_text(text, encoding='utf-8')
    lines = text.strip().splitlines()
    for line in lines[-20:]:
        print(line)
