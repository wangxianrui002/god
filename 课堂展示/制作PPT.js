// 课堂展示 PPT 的可编辑源文件。运行方法见 README.md。
const pptxgen = require('pptxgenjs');
const path = require('path');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = '北京二手房单价预测综合实验';
pptx.subject = '2017 年北京二手房成交单价预测';
pptx.title = '北京二手房单价预测｜课堂汇报';
pptx.lang = 'zh-CN';
pptx.theme = {
  headFontFace: 'Microsoft YaHei',
  bodyFontFace: 'Microsoft YaHei',
  lang: 'zh-CN',
};

const W = 13.333, H = 7.5;
const C = {
  navy: '193149', navy2: '24445D', ink: '213344', muted: '657785',
  paper: 'F7F6F2', white: 'FFFFFF', blue: '297AB7', blueLight: 'DCECF5',
  orange: 'DB6C43', orangeLight: 'F9E9E0', green: '318578', greenLight: 'DFEFEB',
  border: 'D9DFE1', light: 'EEF1F2', greyBar: 'BBC7CC',
};
const fig = path.join(__dirname, 'assets');
const S = pptx.ShapeType;

function txt(slide, text, x, y, w, h, opts={}) {
  slide.addText(text, {x,y,w,h, margin:0, fontFace:'Microsoft YaHei', color:C.ink,
    fontSize:opts.fontSize || 18, breakLine:false, valign:'mid', ...opts});
}
function rect(slide, x,y,w,h,fill, radius=0, lineColor=fill) {
  slide.addShape(radius ? S.roundRect : S.rect, {x,y,w,h,
    rectRadius:radius, radius, line:{color:lineColor, transparency:lineColor===fill?100:0},
    fill:{color:fill}});
}
function line(slide,x1,y1,x2,y2,color=C.border,width=1) {
  slide.addShape(S.line,{x:x1,y:y1,w:x2-x1,h:y2-y1,line:{color,width}});
}
function base(title, section, num, dark=false) {
  const slide=pptx.addSlide();
  slide.background={color:dark?C.navy:C.paper};
  if (!dark) {
    txt(slide,section.toUpperCase(),0.67,0.33,6,0.25,{fontSize:11,color:C.blue,bold:true,charSpacing:1.5});
    txt(slide,title,0.67,0.69,12,0.59,{fontSize:30.5,bold:true,color:C.navy});
    line(slide,0.67,6.94,12.66,6.94,C.border,0.8);
    txt(slide,'2017 北京二手房成交记录 · 综合实验',0.67,7.04,8,0.21,{fontSize:10.5,color:C.muted});
    txt(slide,String(num).padStart(2,'0'),12.12,7.01,0.52,0.25,{fontSize:11,color:C.muted,align:'right'});
  }
  return slide;
}
function pill(slide,text,x,y,w,fill,color=C.ink) {
  rect(slide,x,y,w,0.35,fill,0.13);
  txt(slide,text,x,y,w,0.35,{fontSize:12.5,color,bold:true,align:'center'});
}
function card(slide,x,y,w,h,fill=C.white) {
  rect(slide,x,y,w,h,fill,0.12);
}
function bullet(slide,n,title,body,x,y,w) {
  rect(slide,x,y,0.40,0.40,C.blue,0.20);
  txt(slide,String(n),x,y,0.40,0.40,{fontSize:15,color:C.white,bold:true,align:'center'});
  txt(slide,title,x+0.58,y-0.02,w-0.58,0.38,{fontSize:19.5,bold:true});
  txt(slide,body,x+0.58,y+0.42,w-0.58,0.63,{fontSize:15.5,color:C.muted,valign:'top',breakLine:false});
}

// 1 封面
{
  const s=base('', '',1,true);
  rect(s,0.71,0.83,0.16,5.83,C.orange);
  txt(s,'综合实验 · 机器学习回归',1.16,1.03,8.8,0.42,{fontSize:19.5,color:'A6CBDD',charSpacing:1.0});
  txt(s,'北京二手房\n单价预测',1.16,1.72,10.9,2.00,{fontSize:51,color:C.white,bold:true,breakLine:false,valign:'top'});
  txt(s,'用房屋属性预测 2017 年每平方米成交价',1.18,4.24,10.9,0.54,{fontSize:24,color:'DFE9EC'});
  line(s,1.18,5.38,12.2,5.38,'6B8795',1);
  txt(s,'43,213 条记录',1.18,5.69,3.9,0.53,{fontSize:23,bold:true,color:C.white});
  txt(s,'3 种回归模型',5.05,5.69,3.8,0.53,{fontSize:23,bold:true,color:C.white});
  txt(s,'独立测试集验证',8.82,5.69,3.5,0.53,{fontSize:23,bold:true,color:C.white});
  txt(s,'数据来源：第三方整理的链家北京二手房成交记录',1.18,6.69,11.0,0.27,{fontSize:11.5,color:'A9C0CB'});
  s.addNotes('讲稿第 1 页：研究目标是预测 2017 年北京二手房成交单价，数据 43,213 条，比较三种回归模型。');
}

// 2 数据与问题定义
{
  const s=base('从成交记录到建模样本','01 / 数据与目标',2);
  const yy=1.72, boxW=3.46, gap=0.80;
  const xs=[0.68,0.68+boxW+gap,0.68+2*(boxW+gap)];
  [['318,851','原始成交记录','跨年份的原始表'],['43,217','筛出 2017 年','统一时间范围'],['43,213','最终建模样本','剔除 4 条明显错误']].forEach((v,i)=>{
    card(s,xs[i],yy,boxW,1.55,C.white);
    txt(s,v[0],xs[i]+0.19,yy+0.20,boxW-0.38,0.58,{fontSize:34,color:i===2?C.orange:C.navy,bold:true});
    txt(s,v[1],xs[i]+0.20,yy+0.86,boxW-0.4,0.31,{fontSize:17.5,bold:true});
    txt(s,v[2],xs[i]+0.20,yy+1.20,boxW-0.4,0.23,{fontSize:12.5,color:C.muted});
  });
  txt(s,'→',4.22,2.19,0.54,0.60,{fontSize:29,color:C.blue,align:'center'});
  txt(s,'→',8.48,2.19,0.54,0.60,{fontSize:29,color:C.blue,align:'center'});
  card(s,0.68,3.65,5.78,2.64,C.blueLight);
  pill(s,'预测目标',0.96,3.92,1.22,C.blue,C.white);
  txt(s,'成交单价',0.96,4.40,4.80,0.57,{fontSize:29,bold:true,color:C.navy});
  txt(s,'单位：元/㎡；连续数值 → 回归问题',0.96,5.17,4.99,0.44,{fontSize:18,color:C.ink});
  card(s,6.73,3.65,5.92,2.64,C.white);
  pill(s,'模型输入',7.02,3.92,1.22,C.green,C.white);
  txt(s,'16 项房屋属性',7.02,4.40,5.1,0.57,{fontSize:29,bold:true,color:C.navy});
  txt(s,'面积、户型、区县、房龄、楼层、装修等',7.02,5.11,5.16,0.39,{fontSize:17.3,color:C.ink});
  txt(s,'总价与小区均价不入模，避免价格信息泄露',7.02,5.62,5.10,0.40,{fontSize:14.6,color:C.orange,bold:true});
  s.addNotes('讲稿第 2 页：原始 318,851 条，只保留 2017 年 43,217 条，再剔除 4 条明显错误。目标为单价，16 项特征；总价、小区均价不入模。');
}

// 3 单价分布
{
  const s=base('价格跨度大，高价端有长尾','02 / 数据观察',3);
  card(s,0.68,1.57,8.03,4.98,C.white);
  s.addImage({path:path.join(fig,'fig01_price_distribution.png'),x:0.91,y:1.78,w:7.56,h:4.28,
    altText:'2017 年二手房单价分布直方图'});
  txt(s,'清洗后 43,213 条成交记录',1.00,6.15,7.40,0.22,{fontSize:12,color:C.muted});
  card(s,8.99,1.57,3.66,4.98,C.navy);
  txt(s,'62,332',9.29,1.98,3.05,0.72,{fontSize:37,bold:true,color:C.white});
  txt(s,'单价中位数  元/㎡',9.29,2.75,2.95,0.33,{fontSize:16.5,color:'CFDCE3'});
  line(s,9.29,3.31,12.28,3.31,'6A8693',0.9);
  txt(s,'约 4.9 万—8.1 万',9.29,3.70,3.04,0.60,{fontSize:23,bold:true,color:C.white});
  txt(s,'中间 50% 房源的单价范围',9.29,4.41,2.97,0.65,{fontSize:16,color:'CFDCE3',valign:'top'});
  txt(s,'因此还要单独检查高价房的预测误差',9.29,5.55,2.99,0.55,{fontSize:15,color:'F2C1AA',valign:'top'});
  s.addNotes('讲稿第 3 页：中位数 62,332 元每平米，四分位区间 48,997 到 80,788。高价尾部需要后续误差分析。');
}

// 4 区县差异
{
  const s=base('区县差异是最突出的价格信号','03 / 数据观察',4);
  card(s,0.68,1.57,8.22,4.98,C.white);
  txt(s,'部分区县的成交单价中位数',1.00,1.86,7.66,0.37,{fontSize:19,bold:true});
  const rows=[['房山',38468],['顺义',41317],['昌平',45168],['朝阳',66425],['海淀',84445],['西城',106015]];
  const max=110000, startY=2.37;
  rows.forEach((r,i)=>{
    const y=startY+i*0.62;
    txt(s,r[0],1.03,y,0.79,0.31,{fontSize:16.3,color:C.ink});
    rect(s,1.94,y+0.04,5.18*r[1]/max,0.25,i===5?C.orange:(i===0?C.greyBar:C.blue),0.05);
    txt(s,(r[1]/10000).toFixed(2)+' 万',7.38,y-0.01,1.18,0.34,{fontSize:15.3,align:'right',color:i===5?C.orange:C.ink,bold:i===5});
  });
  txt(s,'仅展示 6 个区县；中位数按区县分别计算',1.00,6.17,7.65,0.22,{fontSize:12,color:C.muted});
  card(s,9.18,1.57,3.47,2.20,C.navy);
  txt(s,'2.76×',9.47,1.96,2.88,0.73,{fontSize:38,bold:true,color:C.white});
  txt(s,'西城 / 房山中位数',9.47,2.87,2.92,0.31,{fontSize:16.5,color:'D0DDE3'});
  card(s,9.18,4.00,3.47,2.55,C.orangeLight);
  txt(s,'62.7%',9.47,4.38,2.88,0.72,{fontSize:37,bold:true,color:C.orange});
  txt(s,'按区县分组的方差解释率 η²',9.47,5.28,2.88,0.62,{fontSize:16,color:C.ink,valign:'top'});
  txt(s,'统计关联，不代表因果比例',9.47,6.04,2.88,0.25,{fontSize:12.7,color:C.muted});
  s.addNotes('讲稿第 4 页：西城中位数 106,015，房山 38,468，比值 2.76。区县的组间方差解释率 62.7%，不能解释为因果。');
}

// 5 模型架构与验证
{
  const s=base('模型架构：16 项房屋属性如何变成单价','04 / 模型架构',5);
  card(s,0.68,1.83,2.22,2.78,C.navy);
  txt(s,'输入',0.96,2.12,1.64,0.36,{fontSize:16.5,color:'C8DAE2',bold:true});
  txt(s,'16 项',0.96,2.62,1.65,0.65,{fontSize:35,color:C.white,bold:true});
  txt(s,'面积 · 房龄 · 区县\n户型 · 装修等',0.96,3.49,1.73,0.81,{fontSize:16,color:C.white,breakLine:false,valign:'top'});
  txt(s,'→',2.95,2.89,0.37,0.47,{fontSize:26,color:C.blue,align:'center'});
  card(s,3.37,1.83,5.13,1.24,C.blueLight);
  txt(s,'数值特征',3.64,2.06,1.49,0.34,{fontSize:18.5,bold:true,color:C.navy});
  txt(s,'缺失值取中位数  →  标准化',5.15,2.06,3.10,0.36,{fontSize:17.2,color:C.ink});
  card(s,3.37,3.37,5.13,1.24,C.greenLight);
  txt(s,'类别特征',3.64,3.60,1.49,0.34,{fontSize:18.5,bold:true,color:C.navy});
  txt(s,'众数填补  →  One-hot 编码',5.15,3.60,3.13,0.36,{fontSize:16.6,color:C.ink});
  txt(s,'→',8.58,2.89,0.37,0.47,{fontSize:26,color:C.blue,align:'center'});
  card(s,9.02,1.83,3.64,2.78,C.white);
  txt(s,'回归模型',9.32,2.12,2.97,0.36,{fontSize:18.5,bold:true,color:C.navy});
  txt(s,'线性回归\nKNN 回归\n决策树回归',9.32,2.72,2.96,1.24,{fontSize:20,color:C.ink,breakLine:false,valign:'top'});
  pill(s,'输出：预测单价 元/㎡',9.32,4.10,2.95,C.orangeLight,C.orange);
  txt(s,'训练与评估',0.70,4.94,4.01,0.38,{fontSize:20.5,bold:true});
  card(s,0.68,5.39,11.98,0.94,C.white);
  txt(s,'80% 训练集 · 34,570 条',0.95,5.64,3.21,0.37,{fontSize:17.4,bold:true,align:'center'});
  txt(s,'→',4.27,5.61,0.41,0.40,{fontSize:23,color:C.blue,align:'center'});
  txt(s,'训练集内 5 折 CV 选模',4.72,5.64,3.96,0.37,{fontSize:17.4,bold:true,align:'center'});
  txt(s,'→',8.79,5.61,0.41,0.40,{fontSize:23,color:C.blue,align:'center'});
  txt(s,'20% 测试集 · 8,643 条',9.15,5.64,3.22,0.37,{fontSize:17.4,bold:true,align:'center'});
  txt(s,'预处理在每折训练管线内重新拟合，避免验证数据提前影响模型；另设均值基线作参照。',0.70,6.51,11.91,0.30,{fontSize:13.9,color:C.muted});
  s.addNotes('讲稿第 5 页：同样的 16 项输入先按数值和类别分开预处理。填补、标准化、One-hot 编码都在 Pipeline 内，交叉验证每折仅用训练折拟合。训练集 34,570，测试集 8,643。');
}

// 6 三种模型原理
{
  const s=base('三种模型如何给出房价预测','05 / 模型原理',6);
  const cards=[
    {x:0.68,w:3.72,color:C.blue,title:'线性回归',tag:'加权求和',body:'为处理后的各个特征学习权重，叠加得到预测单价。',foot:'优点：简洁，是效果参照。'},
    {x:4.81,w:3.72,color:C.green,title:'KNN 回归',tag:'相似房源',body:'在特征空间找最近的 5 套训练房源，取它们的单价均值。',foot:'限制：距离容易受特征表示影响。'},
    {x:8.94,w:3.72,color:C.orange,title:'决策树回归',tag:'条件划分',body:'按区县、面积等条件逐层切分；叶节点房源的均价作为预测。',foot:'优点：可学习非线性和特征组合。'},
  ];
  cards.forEach(v=>{
    card(s,v.x,1.63,v.w,4.94,C.white);
    rect(s,v.x+0.27,1.95,0.15,0.52,v.color);
    txt(s,v.title,v.x+0.55,1.96,2.92,0.48,{fontSize:24,bold:true});
    pill(s,v.tag,v.x+0.27,2.57,1.44,v.color,C.white);
    txt(s,v.body,v.x+0.27,3.17,3.13,1.11,{fontSize:17.5,color:C.ink,valign:'top'});
    line(s,v.x+0.27,4.61,v.x+3.43,4.61,C.border,1);
    txt(s,v.foot,v.x+0.27,4.88,3.13,0.96,{fontSize:16.2,color:C.muted,valign:'top'});
  });
  txt(s,'ŷ = β₀ + Σ βᵢxᵢ',1.22,6.11,2.63,0.30,{fontSize:16.5,color:C.blue,bold:true,align:'center'});
  txt(s,'k = 5',5.86,6.11,1.45,0.30,{fontSize:17,color:C.green,bold:true,align:'center'});
  txt(s,'深度 ≤ 12 · 叶节点 ≥ 20',9.24,6.11,3.16,0.30,{fontSize:14.5,color:C.orange,bold:true,align:'center'});
  s.addNotes('讲稿第 6 页：线性回归是加权求和；KNN 使用处理后的特征距离找 5 套近邻并取单价均值；决策树按条件切分，叶节点输出房源均价。三者使用相同输入与数据划分。');
}

// 7 量化成绩
{
  const s=base('决策树在相同测试集上表现最好','06 / 模型结果',7);
  card(s,0.68,1.57,7.07,4.98,C.white);
  txt(s,'测试集 R²',1.00,1.89,6.40,0.36,{fontSize:20,bold:true});
  const rows=[['均值基线',0,C.greyBar],['KNN',0.6078,C.green],['线性回归',0.6879,C.blue],['决策树',0.7199,C.orange]];
  rows.forEach((r,i)=>{
    let y=2.58+i*0.80;
    txt(s,r[0],1.00,y,1.34,0.36,{fontSize:16.5,bold:i===3});
    rect(s,2.45,y+0.04,4.00*Math.max(r[1],0)/0.75,0.28,r[2],0.05);
    txt(s,i===0?'≈ 0':r[1].toFixed(3),6.42,y-0.01,0.94,0.38,{fontSize:18,color:i===3?C.orange:C.ink,bold:i===3,align:'right'});
  });
  txt(s,'R² 越高越好；均值基线约为 0',1.00,6.11,6.30,0.25,{fontSize:12.5,color:C.muted});
  card(s,8.03,1.57,4.62,3.08,C.navy);
  txt(s,'决策树',8.35,1.88,3.9,0.35,{fontSize:20,bold:true,color:'C9DAE3'});
  txt(s,'0.711',8.35,2.45,3.85,0.76,{fontSize:37,bold:true,color:C.white});
  txt(s,'训练集 5 折 CV R²',8.35,3.22,3.86,0.30,{fontSize:16,color:'C9DAE3'});
  txt(s,'测试 R²  0.720',8.35,3.79,3.85,0.42,{fontSize:21,color:C.white,bold:true});
  card(s,8.03,4.91,4.62,1.64,C.orangeLight);
  txt(s,'9,462',8.35,5.18,3.89,0.60,{fontSize:32,bold:true,color:C.orange});
  txt(s,'测试 MAE · 元/㎡',8.35,5.89,3.89,0.32,{fontSize:15.5,color:C.ink});
  txt(s,'比线性回归少 791 元/㎡；比均值基线少 10,037 元/㎡',0.72,6.62,11.90,0.25,{fontSize:12.8,color:C.muted});
  s.addNotes('讲稿第 7 页：决策树 CV R² 0.711、测试 R² 0.720、MAE 9,462。线性回归测试 R² 0.688、MAE 10,253；KNN 测试 R² 0.608。');
}

// 8 误差
{
  const s=base('高价房更难预测，整体分数不能代替分组检查','07 / 误差分析',8);
  card(s,0.68,1.57,5.65,4.98,C.white);
  s.addImage({path:path.join(fig,'fig04_predictions.png'),x:1.20,y:1.81,w:4.54,h:4.18,
    altText:'决策树真实单价与预测单价散点图'});
  txt(s,'对角线：预测价 = 实际价',1.13,6.16,4.95,0.23,{fontSize:12,color:C.muted,align:'center'});
  card(s,6.61,1.57,6.04,4.98,C.white);
  txt(s,'按真实单价分组的测试集误差',6.92,1.89,5.40,0.39,{fontSize:19.5,bold:true});
  txt(s,'最低 25%',6.92,2.58,2.00,0.33,{fontSize:17});
  txt(s,'7,490',9.64,2.57,2.50,0.36,{fontSize:21,bold:true,align:'right'});
  rect(s,6.92,3.03,2.38,0.22,C.blue,0.05);
  txt(s,'最高 25%',6.92,3.61,2.00,0.33,{fontSize:17});
  txt(s,'14,502',9.64,3.60,2.50,0.36,{fontSize:21,bold:true,color:C.orange,align:'right'});
  rect(s,6.92,4.06,4.61,0.22,C.orange,0.05);
  txt(s,'MAE · 元/㎡',6.92,4.49,5.18,0.25,{fontSize:12.5,color:C.muted});
  line(s,6.92,4.96,12.25,4.96,C.border,0.9);
  txt(s,'高价组平均低估',6.92,5.27,2.80,0.34,{fontSize:17,color:C.ink});
  txt(s,'8,020 元/㎡',9.23,5.18,3.02,0.47,{fontSize:22.4,bold:true,color:C.orange,align:'right'});
  txt(s,'高价组 MAE 约是低价组的 1.94 倍',6.93,6.09,5.31,0.24,{fontSize:13.8,color:C.muted});
  s.addNotes('讲稿第 8 页：最高单价四分位 MAE 14,502，最低为 7,490，高价组平均低估 8,020。整体 R² 不能代表所有价格段。');
}

// 9 演示
{
  const s=base('现场演示：先看真实测试样本，再改输入','08 / 演示',9);
  card(s,0.68,1.57,11.98,3.38,C.white);
  const tx=[1.00,3.43,5.29,7.70,10.07];
  [['样例',tx[0],2.16],['面积',tx[1],1.40],['真实单价',tx[2],2.03],['预测单价',tx[3],2.02],['误差',tx[4],2.02]].forEach(v=>
    txt(s,v[0],v[1],1.95,v[2],0.34,{fontSize:15.5,color:C.muted,bold:true,align:v[0]==='样例'?'left':'right'}));
  line(s,1.00,2.42,12.30,2.42,C.border,1);
  const cases=[['朝阳两居','98.86 ㎡','51,791','60,023','+8,232'],['西城两居','83.23 ㎡','114,983','106,868','−8,115'],['房山两居','85.00 ㎡','44,471','44,429','−42']];
  cases.forEach((r,i)=>{
    const y=2.67+i*0.68;
    if(i===1) rect(s,0.90,y-0.07,11.46,0.59,C.orangeLight,0.07);
    [2.16,1.40,2.03,2.02,2.02].forEach((w,j)=>txt(s,r[j],tx[j],y,w,0.36,
      {fontSize:j===0?17:16.3,bold:j===0||j===4,color:i===1&&j===4?C.orange:C.ink,align:j===0?'left':'right'}));
  });
  txt(s,'单位：元/㎡；误差 = 预测 − 真实。三条均来自固定测试集，仅作示例。',1.00,4.66,11.20,0.20,{fontSize:12.3,color:C.muted});
  card(s,0.68,5.20,11.98,1.35,C.blueLight);
  txt(s,'演示路径',0.98,5.46,1.46,0.35,{fontSize:18.3,bold:true,color:C.navy});
  txt(s,'朝阳 → 西城 → 房山 → 将房山区县改成朝阳并重新预测',2.43,5.43,9.83,0.43,{fontSize:17.6,bold:true,color:C.navy});
  txt(s,'网页输入  →  本机 Python API  →  决策树 Pipeline  →  图形化结果',0.98,6.08,10.94,0.25,{fontSize:13.3,color:C.muted});
  txt(s,'修改输入后不存在对应的真实成交价，只能观察模型输出如何变化。',0.70,6.63,11.89,0.24,{fontSize:12.9,color:C.muted});
  s.addNotes('讲稿第 9 页：现场打开 现场演示.py 启动的本机网页。选择朝阳、西城、房山三个固定测试样本，再把房山区县改成朝阳并计算。网页通过 API 调用模型；修改后没有真实成交价。');
}

// 10 结论
{
  const s=base('','',10,true);
  txt(s,'结论与下一步',0.74,0.61,11.87,0.70,{fontSize:34,bold:true,color:C.white});
  txt(s,'三种模型中，决策树在这份数据上表现最好',0.76,1.40,11.70,0.50,{fontSize:22,color:'C8DAE2'});
  card(s,0.75,2.28,5.58,1.71,C.navy2);
  txt(s,'0.720',1.06,2.60,4.92,0.70,{fontSize:43,bold:true,color:C.white});
  txt(s,'测试集 R²',1.07,3.40,4.65,0.29,{fontSize:16,color:'C8DAE2'});
  card(s,6.63,2.28,5.95,1.71,C.navy2);
  txt(s,'9,462',6.97,2.60,5.27,0.70,{fontSize:43,bold:true,color:'F3BA9D'});
  txt(s,'测试集 MAE · 元/㎡',6.98,3.40,4.90,0.29,{fontSize:16,color:'C8DAE2'});
  txt(s,'解释结果时要保留的边界',0.77,4.49,10.59,0.41,{fontSize:21,bold:true,color:C.white});
  const caveats=[
    '数据为 2017 年成交记录，不能作为今天的报价',
    '高价房误差更大，平均预测单价偏低',
    '随机划分可能偏乐观；后续应按小区或区域评估',
  ];
  caveats.forEach((v,i)=>{
    rect(s,0.81,5.12+i*0.48,0.22,0.22,i===1?C.orange:C.blue,0.11);
    txt(s,v,1.22,5.05+i*0.48,11.10,0.38,{fontSize:17.2,color:'E0EAEE'});
  });
  txt(s,'谢谢聆听',10.06,7.03,2.45,0.25,{fontSize:13.5,color:'A9C0CB',align:'right'});
  s.addNotes('讲稿第 10 页：三种模型中决策树最佳，测试 R² 0.720、MAE 9,462。但数据仅 2017，高价房误差较大，随机划分可能偏乐观。');
}

pptx.writeFile({fileName:path.join(__dirname,'北京二手房单价预测_课堂汇报.pptx')});
