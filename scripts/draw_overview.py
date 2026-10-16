"""Draw a compact, editable research-method schematic in SVG."""
from __future__ import annotations
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'assets/figures/overview.svg'


def txt(x,y,s,size=16,color='#172b4d',bold=False,anchor='start'):
    s=s.replace('&','&amp;').replace('<','&lt;')
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-family="Arial, Helvetica, sans-serif" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{s}</text>'


def line(x,y,x2,y2):
    return f'<path d="M{x} {y} L{x2} {y2}" stroke="#6b8198" stroke-width="2.8" fill="none" marker-end="url(#arrow)"/>'


def rect(x,y,w,h,fill='#ffffff',border='#dae3ef'):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{border}" stroke-width="1.7"/>'

parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="302" viewBox="0 0 1100 302">',
       '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#6b8198"/></marker></defs>',
       '<rect width="1100" height="302" rx="18" fill="#f6f9fd"/>',
       rect(25,40,194,219), rect(255,40,194,219),rect(487,40,206,219),rect(733,40,342,219),
       txt(122,77,'Input image',18,bold=True,anchor='middle'),
       txt(352,77,'Optimize noise',18,bold=True,anchor='middle'),
       txt(590,77,'Frozen layers',18,bold=True,anchor='middle'),
       txt(904,77,'CID = tolerated noise entropy',18,bold=True,anchor='middle')]
# Create 6x6 multichannel colors and geometry for synthetic input schematic
for i in range(6):
 for j in range(6):
  d=(i-2.5)**2+(j-2.5)**2
  color='#d76662' if d<=4 else '#d9e1d5' if (i+j)%2==0 else '#b9d2c7'
  parts.append(f'<rect x="{89+j*11}" y="{113+i*11}" width="11" height="11" fill="{color}"/>')
parts += [txt(122,212,'x',20,bold=True,anchor='middle'), txt(122,239,'Observed pixels',12,'#61758c',anchor='middle')]
# sigma cells
rng=np.random.default_rng(9)
for i in range(6):
 for j in range(6):
  val=float(np.clip((i+j)/10 + rng.normal(0,.12),0,1))
  # manual palette cyan blue
  r=int(221-145*val);g=int(237-76*val);b=int(243-27*val)
  parts.append(f'<rect x="{317+j*11}" y="{114+i*11}" width="11" height="11" fill="#{r:02x}{g:02x}{b:02x}"/>')
parts += [txt(352,212,'σ × ε',19,bold=True,anchor='middle'),txt(352,239,'Feature-preserving',12,'#61758c',anchor='middle')]
# network layers
for i,y in enumerate([115,152,189]):
 parts += [rect(523,y,135,27,'#e8f0f9','#adc0d8'),txt(590,y+19,'h'+str(i+1)+'(x)',14,anchor='middle')]
# heatmap cards
for index,x in enumerate([781,879,977]):
 rng=np.random.default_rng(17+index)
 for i in range(6):
  for j in range(6):
   center=(i-(2.4+.15*index))**2+(j-(2.8-.1*index))**2
   val=float(np.clip(.25+.55/(1+center/4)+rng.normal(0,.14),0,1))
   r=int(248-181*val);g=int(235-99*val);b=int(205+30*val)
   parts.append(f'<rect x="{x+j*10}" y="{115+i*10}" width="10" height="10" fill="#{r:02x}{g:02x}{b:02x}"/>')
 parts.append(txt(x+30,204,['Early','Middle','Late'][index],12,'#61758c',anchor='middle'))
parts += [txt(904,239,'Compare across representation depth',12,'#61758c',anchor='middle'),
          line(219,149,251,149),line(449,149,482,149),line(694,149,729,149),'</svg>']
OUT.write_text('\n'.join(parts)+'\n')
print(OUT)

if __name__=='__main__':
 OUT.write_text('\n'.join(parts)+'\n')
 print(OUT)
