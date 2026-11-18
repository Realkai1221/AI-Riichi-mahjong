from pathlib import Path
import subprocess
renderer='/Users/example/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm'
for page in (81,82,83,84,85,90):
    subprocess.run([renderer,'-f',str(page),'-l',str(page),'-r','180','-singlefile','-png','Books/精修版-Seventh9先生の麻雀教室(79博客).pdf',f'Books/stage3d/evidence/seventh9-page-{page:03d}'],check=True)
