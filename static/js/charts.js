/**
 * static/js/charts.js
 * Chart.js risk gauge for the results page.
 *
 * Renders a half-doughnut (semi-circle) gauge chart showing the fused CVD
 * risk score as a coloured needle-style indicator.
 *
 * Colour segments:
 *   Green  (0.00 – 0.33) → Low Risk
 *   Amber  (0.33 – 0.67) → Moderate Risk
 *   Red    (0.67 – 1.00) → High Risk
 *
 * Usage (called from results.html):
 *   initRiskGauge('risk-gauge', 0.72);
 */

/**
 * Initialise the risk gauge on the specified canvas element.
 *
 * @param {string} canvasId   - The id attribute of the <canvas> element.
 * @param {number} fusedScore - The fused risk score, value between 0.0 and 1.0.
 */
function initRiskGauge(canvasId, fusedScore) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    // Clamp score to [0, 1]
    const score = Math.max(0, Math.min(1, fusedScore));

    // -----------------------------------------------------------------------
    // Gauge segment sizes (total = 1.0, rendered as half-doughnut)
    // -----------------------------------------------------------------------
    const lowEnd      = 0.33;
    const moderateEnd = 0.67;
    const highEnd     = 1.00;

    // -----------------------------------------------------------------------
    // Score indicator
    // Score maps to which segment it falls in — colour-coded
    // -----------------------------------------------------------------------
    let needleColour, riskLabel;
    if (score < 0.33) {
        needleColour = '#22c55e';  // green
        riskLabel    = 'Low';
    } else if (score <= 0.67) {
        needleColour = '#f59e0b';  // amber
        riskLabel    = 'Moderate';
    } else {
        needleColour = '#ef4444';  // red
        riskLabel    = 'High';
    }

    // -----------------------------------------------------------------------
    // The gauge is a doughnut with circumference=Math.PI (half circle).
    // We use the 'rotation' option to start at the left (180°).
    // The 3 data slices represent the three risk zones.
    // A 4th invisible slice (same size as total) fills the bottom half.
    // -----------------------------------------------------------------------
    const gaugeData = {
        datasets: [{
            data: [
                lowEnd,               // Low zone (green)
                moderateEnd - lowEnd, // Moderate zone (amber)
                highEnd - moderateEnd,// High zone (red)
                1.0,                  // Hidden bottom half
            ],
            backgroundColor: [
                '#22c55e',
                '#f59e0b',
                '#ef4444',
                'rgba(0,0,0,0)',      // Transparent
            ],
            borderColor: [
                '#16a34a',
                '#d97706',
                '#dc2626',
                'rgba(0,0,0,0)',
            ],
            borderWidth: 1,
            hoverOffset: 0,
        }],
    };

    const chart = new Chart(ctx, {
        type: 'doughnut',
        data: gaugeData,
        options: {
            rotation: -90,              // Start from left (270° = -90°)
            circumference: 180,         // Half circle
            cutout: '70%',
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false },
            },
            animation: {
                animateRotate: true,
                duration: 800,
            },
            responsive: false,
        },
        plugins: [
            // ---------------------------------------------------------------
            // Custom plugin: draw needle and score text on the canvas
            // ---------------------------------------------------------------
            {
                id: 'gaugeNeedle',
                afterDraw(chart) {
                    const { ctx, chartArea } = chart;
                    const { width, height, left, top } = chartArea;

                    // Centre of the half-doughnut
                    const cx = left + width / 2;
                    const cy = top + height;  // Bottom of the chartArea

                    // Needle angle: score maps [0,1] → [180°, 0°] in radians
                    const angle = Math.PI - (score * Math.PI);

                    // Needle length (slightly shorter than outer radius)
                    const outerRadius = Math.min(width, height * 2) / 2;
                    const needleLength = outerRadius * 0.78;

                    // Needle tip coordinates
                    const tipX = cx + needleLength * Math.cos(angle);
                    const tipY = cy - needleLength * Math.sin(angle);

                    ctx.save();

                    // Draw needle line
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.lineTo(tipX, tipY);
                    ctx.strokeStyle = needleColour;
                    ctx.lineWidth = 3;
                    ctx.lineCap = 'round';
                    ctx.stroke();

                    // Draw centre pivot circle
                    ctx.beginPath();
                    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
                    ctx.fillStyle = needleColour;
                    ctx.fill();

                    // Draw score text below the gauge
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';

                    // Large score number
                    ctx.font = 'bold 18px Inter, sans-serif';
                    ctx.fillStyle = '#E2E8F0';
                    ctx.fillText(score.toFixed(3), cx, cy - needleLength * 0.3);

                    // Risk label below the score
                    ctx.font = '600 12px Inter, sans-serif';
                    ctx.fillStyle = needleColour;
                    ctx.fillText(riskLabel + ' Risk', cx, cy - needleLength * 0.3 + 20);

                    ctx.restore();
                },
            },
        ],
    });

    return chart;
}
