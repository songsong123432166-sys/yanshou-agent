import fs from 'node:fs/promises';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const [input, output] = process.argv.slice(2);
const data = JSON.parse(await fs.readFile(input, 'utf8'));
const wb = Workbook.create();
const name = data.dry_run ? '底线返修' : '辅助复核';
const sheet = wb.worksheets.add(name);
sheet.showGridLines = false;
const entries = data.entries;
const bottom = entries.length + 1;
const rows = [['题目 ID', '错误项'], ...entries];
// All identifiers and findings are literal text, including leading = and zeros.
sheet.getRange(`A1:B${bottom}`).setNumberFormat('@');
sheet.getRange(`A1:B${bottom}`).values = rows.map(row => row.map(value =>
  typeof value === 'string' && value.startsWith('=') ? "'" + value : value));
sheet.getRange(`A1:B${bottom}`).format = {
  font: { name: 'Arial', size: 11, color: '#253247' },
  verticalAlignment: 'center',
  rowHeight: 34,
};
sheet.getRange(`A1:A${bottom}`).format.columnWidth = 27;
sheet.getRange(`B1:B${bottom}`).format.columnWidth = 100;
sheet.getRange(`A1:A${bottom}`).format.horizontalAlignment = 'left';
sheet.getRange(`B1:B${bottom}`).format.wrapText = true;
sheet.getRange('A1:B1').format = {
  fill: '#243A57',
  font: { name: 'Arial', size: 11, bold: true, color: '#FFFFFF' },
  horizontalAlignment: 'center', verticalAlignment: 'center', rowHeight: 34,
  borders: { insideVertical: { style: 'thin', color: '#FFFFFF' } },
};
if (entries.length) {
  const table = sheet.tables.add(`A1:B${bottom}`, true, 'CaptionRepair');
  table.showFilterButton = true;
  // Explicit muted banding stays legible without a full body-cell grid.
  for (let i = 0; i < entries.length; i++) {
    const row = i + 2;
    const lines = entries[i][1].split('\n');
    const wrapped = lines.reduce((sum, line) => sum + Math.max(1, Math.ceil(
      [...line].reduce((width, c) => width + (c.charCodeAt(0) > 255 ? 2 : 1), 0) / 90)), 0);
    sheet.getRange(`A${row}:B${row}`).format.rowHeight = Math.min(409, Math.max(34, wrapped * 18 + 14));
    sheet.getRange(`A${row}:B${row}`).format.fill = i % 2 === 1 ? '#F0F4F8' : '#FFFFFF';
    sheet.getRange(`A${row}`).format.font.bold = true;
  }
}
// A two-column report remains two columns; coverage is labeled outside its rows.
if (!data.complete) {
  const noteRow = bottom + 2;
  sheet.getRange(`B${noteRow}`).values = [['检测或 ID 尚不完整，本表仅含已发现且有 ID 的问题，请同时查看检测说明。']];
  sheet.getRange(`B${noteRow}`).format = { wrapText: true, rowHeight: 42,
    font: { name: 'Arial', size: 11, bold: true, color: '#9C5B14' } };
}
if (!entries.length) {
  sheet.getRange('B3').values = [[data.complete ? '当前已检测范围内无返修项。' : '尚无可列出的返修项；检测未完成不等于通过。']];
  sheet.getRange('B3').format.font = { name: 'Arial', size: 11, color: '#526075' };
}
sheet.freezePanes.freezeRows(1);
sheet.tabColor = '#243A57';
wb.recalculate();
// One bounded inspect verifies the live workbook values without printing bodies.
await wb.inspect({ kind: 'table', range: `${name}!A1:B${Math.min(bottom, 5)}`,
  include: 'values', tableMaxRows: 5, tableMaxCols: 2, maxChars: 2000 });
if (data.preview_path) {
  const preview = await wb.render({ sheetName: name, range: `A1:B${Math.min(bottom, 9)}`, scale: 1.5, format: 'png' });
  await fs.writeFile(data.preview_path, new Uint8Array(await preview.arrayBuffer()));
}
const file = await SpreadsheetFile.exportXlsx(wb);
await file.save(output);
