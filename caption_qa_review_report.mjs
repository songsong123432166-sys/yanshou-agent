import fs from 'node:fs/promises';
import {Workbook,SpreadsheetFile,FileBlob} from '@oai/artifact-tool';
const [input,output,preview]=process.argv.slice(2);
if(input==='--preview'){
  const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(output));
  const image=await wb.render({sheetName:'人工审核',range:'B1:G3',scale:1,format:'png'});
  await fs.writeFile(preview,new Uint8Array(await image.arrayBuffer()));
}else{
  const data=JSON.parse(await fs.readFile(input,'utf8'));
  const wb=Workbook.create(),sheet=wb.worksheets.add('人工审核'),bottom=data.rows.length+1;
  sheet.showGridLines=false;
  const range=sheet.getRange(`A1:H${bottom}`);
  range.setNumberFormat('@');
  range.values=[data.headers,...data.rows].map(row=>row.map(v=>v.startsWith('=')?"'"+v:v));
  range.format={font:{name:'Arial',size:11,color:'#253247'},verticalAlignment:'top',wrapText:true,rowHeight:120};
  for(const [col,width] of [['A',28],['B',22],['C',18],['D',48],['E',85],['F',70],['G',22],['H',85]])sheet.getRange(`${col}1:${col}${bottom}`).format.columnWidth=width;
  sheet.getRange('A1:H1').format={fill:'#243A57',font:{name:'Arial',size:11,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',verticalAlignment:'center',rowHeight:32};
  if(data.rows.length){
    const table=sheet.tables.add(`A1:H${bottom}`,true,'CaptionHumanReview');table.showFilterButton=true;
    sheet.getRange(`G2:H${bottom}`).format.fill='#FFF0D6';
    sheet.getRange(`G2:G${bottom}`).dataValidation={rule:{type:'list',values:data.decision_labels}};
  }
  sheet.freezePanes.freezeRows(1);wb.recalculate();
  await wb.inspect({kind:'table',range:'人工审核!B1:G3',include:'values',tableMaxRows:3,tableMaxCols:6,maxChars:1000});
  const file=await SpreadsheetFile.exportXlsx(wb);await file.save(output);
}
