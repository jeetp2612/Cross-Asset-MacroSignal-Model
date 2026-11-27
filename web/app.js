async function loadJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error("Could not load " + url);
  return response.json();
}

async function main() {
  const [overview, history, direction] = await Promise.all([
    loadJSON("/api/overview"),
    loadJSON("/api/history"),
    loadJSON("/api/direction"),
  ]);

  document.querySelector("#vix").textContent = overview.vix ?? "--";
  document.querySelector("#curve").textContent = overview.curve ?? "--";
  document.querySelector("#spy").textContent = overview.sp500 ?? "--";
  document.querySelector("#forecastDate").textContent = "Forecast date: " + (overview.cards[0]?.date || "--");

  const wild = overview.cards.filter(row => row.label.includes("Wilder")).length;
  document.querySelector("#mood").textContent = wild
    ? wild + " assets may face wilder-than-usual risk"
    : "Markets show normal risk conditions";

  document.querySelector("#cards").innerHTML = overview.cards.map(row => `
    <article class="card ${row.label.includes("Wilder") ? "high" : ""}">
      <h3>${row.asset}</h3>
      <div class="forecast">${row.forecast.toFixed(2)} <small>% volatility</small></div>
      <div class="label">${row.label}</div>
      <dl>
        <dt>Current risk</dt><dd>${row.current.toFixed(2)}%</dd>
        <dt>Best model</dt><dd>${row.model}</dd>
        <dt>Wild probability</dt><dd>${row.wildProb.toFixed(1)}%</dd>
      </dl>
    </article>
  `).join("");

  const bestScores = {};
  for (const row of overview.scores) {
    if (!bestScores[row.asset] || row.improvement > bestScores[row.asset].improvement) {
      bestScores[row.asset] = row;
    }
  }
  document.querySelector("#scores").innerHTML = Object.values(bestScores).map(row => {
    const width = Math.max(5, Math.min(100, 50 + row.improvement));
    return `
      <div class="score-row">
        <span>${row.asset}</span>
        <div class="bar"><span style="width:${width}%"></span></div>
        <strong>${row.improvement.toFixed(1)}%</strong>
      </div>
    `;
  }).join("");

  drawChart(history);

  const assets = ["SPY", "TLT", "GLD", "UUP"];
  document.querySelector("#direction").innerHTML = assets.map(asset => {
    const rows = direction.filter(row => row.asset === asset);
    const best = rows.sort((a, b) => b.accuracy - a.accuracy)[0];
    return `
      <div class="direction-item">
        <h3>${asset}</h3>
        <p>Best model: <strong>${best.model}</strong></p>
        <p>Average AUC: <strong>${best.accuracy.toFixed(3)}</strong></p>
        <p>Strategy vs buy & hold:
          <strong>${best.strategy.toFixed(2)} vs ${best.buyHold.toFixed(2)}</strong>
        </p>
      </div>
    `;
  }).join("");
}

function drawChart(rows) {
  const canvas = document.querySelector("#chart");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const colors = { SPY: "#326da8", TLT: "#7b61a8", GLD: "#c28b32", UUP: "#4d9b7a" };
  const left = 45, right = 20, top = 20, bottom = 35;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const values = rows.flatMap(row => [row.SPY, row.TLT, row.GLD, row.UUP]).filter(Number.isFinite);
  const min = 0;
  const max = Math.ceil(Math.max(...values) / 10) * 10;

  ctx.strokeStyle = "#e3e9ef";
  ctx.fillStyle = "#7a8795";
  ctx.font = "11px Arial";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const y = top + plotHeight * i / 4;
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(width - right, y);
    ctx.stroke();
    ctx.fillText((max - (max - min) * i / 4).toFixed(0) + "%", left - 8, y + 4);
  }

  Object.keys(colors).forEach(asset => {
    ctx.strokeStyle = colors[asset];
    ctx.lineWidth = 2;
    ctx.beginPath();
    rows.forEach((row, index) => {
      const x = left + plotWidth * index / Math.max(1, rows.length - 1);
      const y = top + plotHeight * (1 - (row[asset] - min) / (max - min));
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  ctx.textAlign = "left";
  Object.entries(colors).forEach(([asset, color], index) => {
    ctx.fillStyle = color;
    ctx.fillRect(left + index * 70, height - 16, 18, 3);
    ctx.fillStyle = "#69788a";
    ctx.fillText(asset, left + index * 70 + 23, height - 10);
  });
}

main().catch(error => {
  document.querySelector("#mood").textContent = "Could not load project data";
  document.querySelector("#cards").textContent = error.message;
});
