/** Minimal safe markdown renderer: headings, bold/italic, lists, tables,
 * code (inline + fenced), links, blockquotes, rules. Everything is escaped
 * first — no raw HTML from the input ever reaches the DOM. */

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inline(s) {
  let out = esc(s);
  out = out.replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  out = out.replace(/\[([^\]]+)\]\(([^)\s]+(?:\([^)]*\)[^)\s]*)*)\)/g, (m, text, url) => {
    const safe = /^(https?:|mailto:|#|\/)/i.test(url) ? url.replace(/&amp;/g, "&") : "#";
    return `<a href="${esc(safe)}" rel="noopener">${text}</a>`;
  });
  return out;
}

function isTableSep(line) {
  return /^\s*\|?[\s:|-]+\|?\s*$/.test(line) && line.includes("-");
}

function splitRow(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
}

/** Render markdown text to an HTML string (input is never trusted). */
export function renderMarkdown(md) {
  const lines = String(md || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let i = 0;
  let listType = null;
  const closeList = () => { if (listType) { html.push(listType === "ul" ? "</ul>" : "</ol>"); listType = null; } };

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim().startsWith("```")) {
      closeList();
      const buf = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) { buf.push(lines[i]); i++; }
      i++;
      html.push(`<pre><code>${esc(buf.join("\n"))}</code></pre>`);
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      closeList();
      const head = splitRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim() !== "") {
        rows.push(splitRow(lines[i])); i++;
      }
      const th = head.map((c) => `<th>${inline(c)}</th>`).join("");
      const trs = rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("");
      html.push(`<table><thead><tr>${th}</tr></thead><tbody>${trs}</tbody></table>`);
      continue;
    }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { closeList(); html.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); i++; continue; }
    if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) { closeList(); html.push("<hr>"); i++; continue; }
    if (line.trim().startsWith(">")) {
      closeList();
      const buf = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) { buf.push(lines[i].trim().replace(/^>\s?/, "")); i++; }
      html.push(`<blockquote>${buf.map(inline).join("<br>")}</blockquote>`);
      continue;
    }
    const ul = line.match(/^\s*[-*]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ul || ol) {
      const type = ul ? "ul" : "ol";
      if (listType !== type) { closeList(); html.push(type === "ul" ? "<ul>" : "<ol>"); listType = type; }
      html.push(`<li>${inline((ul || ol)[1])}</li>`);
      i++; continue;
    }
    if (line.trim() === "") { closeList(); i++; continue; }
    closeList();
    const para = [line];
    i++;
    while (i < lines.length && lines[i].trim() !== "" && !/^(#{1,6})\s/.test(lines[i])
      && !lines[i].trim().startsWith("```") && !/^\s*[-*]\s/.test(lines[i])
      && !/^\s*\d+[.)]\s/.test(lines[i])) { para.push(lines[i]); i++; }
    html.push(`<p>${para.map(inline).join("<br>")}</p>`);
  }
  closeList();
  return html.join("\n");
}
