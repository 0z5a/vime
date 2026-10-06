"""Typeset the audited standalone manuscript as a portable PDF without TeX installation.

The editable LaTeX stays authoritative and is independently compiled by the native editor.
This PDF export uses ReportLab text/tables and Matplotlib's embedded math fonts.
"""
import sys
from pathlib import Path
sys.path.append('/Users/0z5a/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
import hashlib
import html
import io
import json
import re
from PIL import Image as PILImage
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image, MathTextParser
from matplotlib import rcParams
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import SimpleDocTemplate, BaseDocTemplate, PageTemplate, Frame, FrameBreak, Flowable, HRFlowable, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether, PageBreak

rcParams['mathtext.fontset']='stix'

ROOT=Path(__file__).absolute().parent.parent
SOURCE=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'paper/flashrlt.tex'
OUTPUT=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT.parent/'output/pdf/ScaleRLT_v0.1_2026-10-06.pdf'
TMP=ROOT.parent/'tmp/pdfs/scalerlt-v01'
TMP.mkdir(parents=True,exist_ok=True)
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
source=SOURCE.read_text()
GENERATION=Path(sys.argv[3]) if len(sys.argv)>3 else ROOT/'paper/generation.json'
generation=json.loads(GENERATION.read_text())
assert generation['preliminary'] and generation['compilation_status']=='SUCCESS'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==generation['latex_sha256']

WIDTH=243
styles=getSampleStyleSheet()
styles.add(ParagraphStyle('Paper',fontName='Times-Roman',fontSize=9.2,leading=11.6,
    alignment=TA_JUSTIFY,spaceAfter=5.5,splitLongWords=True))
styles.add(ParagraphStyle('PaperTitle',fontName='Times-Bold',fontSize=20,leading=24,
    alignment=TA_CENTER,spaceAfter=12))
styles.add(ParagraphStyle('PaperDate',fontName='Times-Roman',fontSize=10.5,leading=14,
    alignment=TA_CENTER,spaceAfter=17))
styles.add(ParagraphStyle('PaperSection',fontName='Times-Bold',fontSize=11.6,leading=14,
    spaceBefore=10,spaceAfter=5,keepWithNext=True))
styles.add(ParagraphStyle('PaperSubsection',fontName='Times-Bold',fontSize=9.6,leading=12,
    spaceBefore=8,spaceAfter=4,keepWithNext=True))
styles.add(ParagraphStyle('PaperCaption',fontName='Times-Roman',fontSize=8.2,leading=10.2,
    alignment=TA_LEFT,spaceBefore=5,spaceAfter=8))
styles.add(ParagraphStyle('PaperCell',fontName='Times-Roman',fontSize=7.8,leading=9.6,
    alignment=TA_LEFT,splitLongWords=True))
styles.add(ParagraphStyle('PaperHead',parent=styles['PaperCell'],fontName='Times-Bold'))
styles.add(ParagraphStyle('PaperReference',fontName='Times-Roman',fontSize=8.2,leading=10.3,
    alignment=TA_LEFT,spaceAfter=4,splitLongWords=True))

def braced(text,pos):
    while pos<len(text) and text[pos].isspace():pos+=1
    assert pos<len(text) and text[pos]=='{',(text[pos:pos+70],pos)
    level=1;start=pos+1;pos+=1
    while level:
        if text[pos]=='{' and (pos==0 or text[pos-1]!='\\'):level+=1
        if text[pos]=='}' and text[pos-1]!='\\':level-=1
        pos+=1
    return text[start:pos-1],pos

def argument(text,name):
    match=re.search(r'\\'+re.escape(name)+r'\b',text)
    assert match,name
    return braced(text,match.end())[0]

bibkeys=re.findall(r'\\bibitem\{([^}]+)\}',source)
citation={key:str(i+1) for i,key in enumerate(bibkeys)}
labels={};table_count=figure_count=equation_count=0
env_regex=re.compile(r'\\begin\{(equation|align|table\*?|figure\*?|longtable)\}(?:\[[^\]]*\])?')
for match in env_regex.finditer(source):
    env=match.group(1);end=source.index('\\end{'+env+'}',match.end())
    block=source[match.end():end]
    if env.rstrip('*') in ('table','longtable'):table_count+=1;number=table_count
    elif env.rstrip('*')=='figure':figure_count+=1;number=figure_count
    else:equation_count+=1;number=equation_count
    for key in re.findall(r'\\label\{([^}]+)\}',block):labels[key]=str(number)


section_index=0; appendix_index=False
document_source=source[source.index(r'\begin{document}'):]
for match in re.finditer(r'\\appendix\b|\\section\b',document_source):
    if match.group(0)==r'\appendix':
        appendix_index=True;section_index=0;continue
    _,end=braced(document_source,match.end());section_index+=1
    following=re.match(r'\s*\\label\{([^}]+)\}',document_source[end:])
    if following:labels[following.group(1)]=chr(64+section_index) if appendix_index else str(section_index)

math_cache={};math_parser=MathTextParser('path')
def math_asset(expression,size=10.5):
    expression=re.sub(r'\\label\{[^}]*\}','',expression)
    expression=expression.replace('\\nonumber','').replace('&','')
    expression=expression.replace('\\tfrac','\\frac')
    expression=re.sub(r'\\frac([0-9])([0-9])',r'\\frac{\1}{\2}',expression)
    expression=re.sub(r'\{\\rm\s+([^}]+)\}',r'{\\mathrm{\1}}',expression)
    expression=re.sub(r'\\big\b','',expression)
    expression=expression.replace(r'\mathcal R',r'\mathcal{R}').replace(r'\mathsf T',r'\mathsf{T}')
    expression=re.sub(r'\\ge\b',r'\\geq',expression)
    expression=' '.join(expression.split()).strip()
    key=(expression,size)
    if key not in math_cache:
        math='$'+expression+'$'
        prop=FontProperties(family='STIXGeneral',size=size)
        parsed=math_parser.parse(math,dpi=72,prop=prop)
        data=io.BytesIO()
        math_to_image(math,data,prop=prop,dpi=300,format='png',color='#111111')
        path=TMP/('math-'+hashlib.sha256(repr(key).encode()).hexdigest()[:20]+'.png')
        path.write_bytes(data.getvalue())
        with PILImage.open(path) as im:w,h=im.size
        math_cache[key]=(path,w*72/300,h*72/300,float(parsed.depth))
    return math_cache[key]

def inline_math(expression,size=10.5):
    numeric=re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)\\times',expression)
    if numeric:return numeric.group(1)+'&#215;'
    p,w,h,depth=math_asset(expression,size)
    return '<img src="'+str(p)+'" width="'+str(w)+'" height="'+str(h)+'" valign="'+str(-depth)+'"/>'

def text_html(text,size=9.2):
    result=[];i=0
    while i<len(text):
        ch=text[i]
        if ch=='$':
            end=text.index('$',i+1)
            result.append(inline_math(text[i+1:end],size));i=end+1;continue
        if ch!='\\':
            if ch in '{}':i+=1;continue
            result.append(html.escape(ch));i+=1;continue
        if i+1<len(text) and text[i+1] in '%_&#{}$':
            result.append(html.escape(text[i+1]));i+=2;continue
        match=re.match(r'\\([A-Za-z]+)',text[i:])
        if not match:
            if text[i:i+2]=='\\\\':result.append('<br/>');i+=2
            else:i+=1
            continue
        cmd=match.group(1);i+=len(match.group(0))
        if cmd in ('textbf','emph','texttt','code','textsc','cite','ref','eqref','url','href'):
            value,i=braced(text,i)
            if cmd=='textbf':result.append('<b>'+text_html(value,size)+'</b>')
            elif cmd=='emph':result.append('<i>'+text_html(value,size)+'</i>')
            elif cmd in ('texttt','code'):result.append('<font face="Courier" size="'+str(size*.86)+'">'+text_html(value,size*.86)+'</font>')
            elif cmd=='textsc':result.append(text_html(value,size).upper())
            elif cmd=='cite':result.append('['+', '.join(citation[k.strip()] for k in value.split(','))+']')
            elif cmd in ('ref','eqref'):
                assert value in labels,value
                result.append(('('+labels[value]+')') if cmd=='eqref' else labels[value])
            elif cmd=='url':result.append('<link href="'+html.escape(value,quote=True)+'" color="#176b72">'+html.escape(value)+'</link>')
            elif cmd=='href':
                label,i=braced(text,i)
                result.append('<link href="'+html.escape(value,quote=True)+'" color="#176b72">'+text_html(label,size)+'</link>')
        elif cmd in ('label','setlength'):
            _,i=braced(text,i)
            if cmd=='setlength':_,i=braced(text,i)
        elif cmd=='newline':result.append('<br/>')
        elif cmd in ('noindent','centering','small','normalsize','maketitle','appendix','clearpage','ttfamily','raggedright','scriptsize'):pass
        else:raise ValueError('Unhandled prose command: '+cmd)
    return re.sub(r'\s+',' ',''.join(result)).replace('~',' ').replace('---','-').replace('--','-').strip()

story=[];section=0;subsection=0;appendix=False
def prose(text):
    global section,subsection,appendix
    pattern=re.compile(r'\\(section|subsection)\b|\\appendix\b')
    pos=0
    while True:
        m=pattern.search(text,pos)
        piece=text[pos:m.start()] if m else text[pos:]
        for para in re.split(r'\n\s*\n',piece):
            val=text_html(para)
            if val:story.append(Paragraph(val,styles['Paper']))
        if not m:break
        if m.group(0)=='\\appendix':
            story.append(PageBreak());appendix=True;section=0;subsection=0;pos=m.end();continue
        heading,pos=braced(text,m.end())
        if m.group(1)=='section':
            section+=1;subsection=0
            if appendix and heading=='Every planned public solve':story.append(PageBreak())
            number=chr(64+section) if appendix else str(section)
            story.append(Paragraph(number+' '+text_html(heading),styles['PaperSection']))
        else:
            subsection+=1
            number=(chr(64+section) if appendix else str(section))+'.'+str(subsection)
            story.append(Paragraph(number+' '+text_html(heading),styles['PaperSubsection']))

table_number=figure_number=equation_number=0
def equation(block):
    global equation_number
    equation_number+=1
    block=re.sub(r'\\label\{[^}]*\}','',block)
    lines=[line.strip() for line in re.split(r'\\\\',block) if line.strip()]
    if len(lines)==1 and '\\quad' in lines[0] and math_asset(lines[0],11)[1]>WIDTH-28:
        lines=[line.strip() for line in re.split(r',?\s*\\qquad\b|,?\s*\\quad\b',lines[0]) if line.strip()]
    group=[]
    for j,line in enumerate(lines):
        p,w,h,_=math_asset(line,9.6)
        scale=min(1,(WIDTH-28)/w)
        im=Image(str(p),width=w*scale,height=h*scale)
        number=Paragraph('('+str(equation_number)+')' if j==len(lines)-1 else '',styles['PaperCell'])
        row=Table([[im,number]],colWidths=[WIDTH-28,28],hAlign='CENTER')
        row.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(0,0),(0,0),'CENTER'),('ALIGN',(1,0),(1,0),'RIGHT'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)]))
        group.append(row)
    story.append(KeepTogether(group+[Spacer(1,7)]))

def clean_rules(text):
    return re.sub(r'\\(?:toprule|midrule|bottomrule|endfirsthead|endhead)\b','',text)

def table(block,long=False):
    global table_number
    table_number+=1
    cap=argument(block,'caption')
    caption=Paragraph('<b>Table '+str(table_number)+'.</b> '+text_html(cap,8.2),styles['PaperCaption'])
    if long:
        m=re.search(r'\\caption\b',block);_,cap_end=braced(block,m.end())
        header=re.search(r'\\toprule(.*?)\\midrule',block,re.S).group(1)
        data=header+'\\\\'+block[block.index('\\endhead')+len('\\endhead'):]
        data=re.sub(r'^\s*\{[^}]*\}','',data)
    else:
        start=block.index(r'\begin{tabular}')+len(r'\begin{tabular}')
        _,start=braced(block,start)
        data=block[start:block.index(r'\end{tabular}',start)]
    rows=[]
    for line in re.split(r'\\\\',clean_rules(data)):
        line=line.strip()
        if line:rows.append([cell.strip() for cell in line.split('&')])
    columns=len(rows[0]);assert all(len(row)==columns for row in rows),rows
    vals=[];widths=[0]*columns
    for r,row in enumerate(rows):
        out=[]
        for c,cell in enumerate(row):
            formatted=text_html(cell,7.8)
            out.append(Paragraph(formatted,styles['PaperHead'] if r==0 else styles['PaperCell']))
            plain=re.sub(r'<[^>]+>','',formatted)
            widths[c]=max(widths[c],min(100,max(30,stringWidth(html.unescape(plain),'Times-Roman',7.8)+8)))
        vals.append(out)
    factor=WIDTH/sum(widths)
    widths=[w*factor for w in widths]
    tab=Table(vals,colWidths=widths,repeatRows=1,hAlign='LEFT')
    tab.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.8,colors.HexColor('#222222')),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#777777')),('LINEBELOW',(0,-1),(-1,-1),.7,colors.HexColor('#222222')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),3.2),('BOTTOMPADDING',(0,0),(-1,-1),3.2),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]))
    if long:story.extend([caption,tab,Spacer(1,11)])
    else:story.append(KeepTogether([caption,tab,Spacer(1,11)]))

class ExecutionDiagram(Flowable):
    def __init__(self):
        super().__init__();self.width=504;self.height=112
    def draw(self):
        c=self.canv;c.saveState();c.scale(504/480,1)
        c.setStrokeColor(colors.HexColor('#126974'));c.setLineWidth(.8)
        labels=[('Committed policy','Rollout + reward'),('Shared prefix','All loop boundaries'),('Suffix waves','Readout + adjoints'),('Logical update','Adam + publication')]
        for x,(top,bottom) in zip([0,124,248,372],labels):
            c.setFillColor(colors.HexColor('#edf5f6'));c.rect(x,64,104,36,fill=1)
            c.setFillColor(colors.HexColor('#162739'));c.setFont('Helvetica-Bold',8.2);c.drawCentredString(x+52,84,top)
            c.setFont('Helvetica',7.8);c.drawCentredString(x+52,72,bottom)
        c.setFillColor(colors.HexColor('#162739'));c.setStrokeColor(colors.HexColor('#162739'))
        for x in [106,230,354]:
            c.line(x,82,x+16,82);c.line(x+16,82,x+12,85);c.line(x+16,82,x+12,79)
        c.line(300,61,300,43);c.line(300,43,176,43);c.line(176,43,176,61)
        c.line(176,61,173,57);c.line(176,61,179,57)
        c.setFont('Helvetica',8);c.drawCentredString(240,28,'Summed boundary gradients')
        c.setFillColor(colors.HexColor('#edf5f6'));c.rect(0,1,476,20,stroke=0,fill=1)
        c.setFillColor(colors.HexColor('#5f6d78'));c.setFont('Helvetica',8)
        c.drawCentredString(238,8,'Policy/latent identity  |  Stage KV + rematerialization  |  Verified durable state')
        c.restoreState()

def figure(block):
    global figure_number
    figure_number+=1
    assert 'fig:system' in block
    # The full-width design figure appears below the first-page abstract.

def bibliography(block):
    group=[Paragraph('References',styles['PaperSection'])]
    pieces=re.split(r'\\bibitem\{([^}]+)\}',block)[1:]
    for i in range(0,len(pieces),2):
        key,piece=pieces[i:i+2]
        group.append(Paragraph('['+citation[key]+'] '+text_html(piece,8.2),styles['PaperReference']))
    story.extend(group)


abstract=source.split(r'\begin{minipage}{0.94\textwidth}\small',1)[1].split(r'\end{minipage}',1)[0]
figblock=re.search(r'\\begin\{figure\*\}.*?\\end\{figure\*\}',source,re.S).group(0)
styles.add(ParagraphStyle('PaperAbstract',fontName='Times-Roman',fontSize=9.3,leading=11.8,alignment=TA_JUSTIFY,spaceAfter=9))
front=[Paragraph(text_html(argument(source,'title')),styles['PaperTitle']),
       Paragraph('0z5a | Research draft v0.1 | October 6, 2026',styles['PaperDate']),
       Paragraph(text_html(abstract,9.3),styles['PaperAbstract']),
       HRFlowable(width=504,thickness=.8,color=colors.HexColor('#126974')),Spacer(1,9),
       ExecutionDiagram(),Paragraph('<b>Figure 1.</b> '+text_html(argument(figblock,'caption'),8.2),styles['PaperCaption'])]
front_height=sum(f.wrap(504,1000)[1]+f.getSpaceBefore()+f.getSpaceAfter() for f in front)+8
assert front_height<520,front_height
story.extend(front+[FrameBreak()])
body=source[source.index(r'\section{Introduction}'):].split(r'\end{document}',1)[0]
body=body.replace(r'\begin{quote}','').replace(r'\end{quote}','')
pattern=re.compile(r'\\begin\{(abstract|equation|align|table\*?|figure\*?|thebibliography|longtable)\}(?:\[[^\]]*\])?')
pos=0
while True:
    m=pattern.search(body,pos)
    if not m:prose(body[pos:]);break
    prose(body[pos:m.start()])
    env=m.group(1);end=body.index('\\end{'+env+'}',m.end())
    block=body[m.end():end]
    if env=='abstract':
        story.append(Paragraph('Abstract',styles['PaperSubsection']));prose(block)
    elif env in ('equation','align'):equation(block)
    elif env.rstrip('*') in ('table','longtable'):table(block,long=env=='longtable')
    elif env.rstrip('*')=='figure':figure(block)
    elif env=='thebibliography':bibliography(block)
    pos=end+len('\\end{'+env+'}')

def page(canvas,doc):
    canvas.saveState()
    canvas.setTitle(argument(source,'title'))
    canvas.setSubject('ScaleRLT v0.1: audited component evidence; full RL and reward convergence pending.')
    canvas.setAuthor('0z5a')
    if doc.page>1:
        canvas.setFont('Times-Roman',8.2);canvas.setFillColor(colors.HexColor('#666666'))
        canvas.drawString(54,757,'ScaleRLT | Research draft v0.1')
        canvas.line(54,750,558,750)
    canvas.setFont('Times-Roman',9);canvas.setFillColor(colors.HexColor('#444444'))
    canvas.drawCentredString(306,35,str(doc.page))
    canvas.restoreState()

def column(x,y,height,name,width=243):
    return Frame(x,y,width,height,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,id=name)
body_top=738;bottom=50
first_bottom_top=body_top-front_height-10
first=[column(54,body_top-front_height,front_height,'front',504),column(54,bottom,first_bottom_top-bottom,'first-left'),column(310,bottom,first_bottom_top-bottom,'first-right')]
normal=[column(54,bottom,body_top-bottom,'left'),column(310,bottom,body_top-bottom,'right')]
doc=BaseDocTemplate(str(OUTPUT),pagesize=(612,792),leftMargin=54,rightMargin=54,topMargin=54,bottomMargin=50,title=argument(source,'title'),author='0z5a')
doc.addPageTemplates([PageTemplate(id='first',frames=first,onPage=page,autoNextPageTemplate='normal'),PageTemplate(id='normal',frames=normal,onPage=page)])
doc.build(story)
from pypdf import PdfReader
reader=PdfReader(OUTPUT)
texts=[p.extract_text() for p in reader.pages]
text='\n'.join(texts)
for needle in ['2.963','3.675','48.44','1.1424','26/32','Reward and Scaled-RL Qualification','Not run']:
    assert needle in text,needle
assert table_number==12 and figure_number==1
receipt={'pdf':str(OUTPUT),'pdf_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    'source':str(SOURCE),'source_sha256':generation['latex_sha256'],'pages':len(reader.pages),
    'tables':table_number,'figures':figure_number,'equation_groups':equation_number,
    'math_expressions':len(math_cache),'export_method':'ReportLab two-column typesetting from compiled LaTeX; embedded STIX math at 300 dpi',
    'preliminary':True,'reward_convergence':'NOT_RUN','visual_verification':'PENDING'}
(ROOT/'paper/pdf-export.json').write_text(json.dumps(receipt,indent=2)+'\n')
(TMP/'extracted-text.txt').write_text(text)
print(json.dumps(receipt,indent=2))
