const companyName = document.getElementById('companyName');
const currentPrice = document.getElementById('currentPrice');
const epsInput = document.getElementById('eps');
const growthRateInput = document.getElementById('growthRate');
const targetPEInput = document.getElementById('targetPE');
const discountRateInput = document.getElementById('discountRate');
const sharesOutstandingInput = document.getElementById('sharesOutstanding');
const calculateBtn = document.getElementById('calculateBtn');
const peValue = document.getElementById('peValue');
const dcfValue = document.getElementById('dcfValue');
const fairValue = document.getElementById('fairValue');
const valuationLabel = document.getElementById('valuationLabel');

function formatCurrency(value) {
  return new Intl.NumberFormat('vi-VN', {
    style: 'currency',
    currency: 'VND',
    maximumFractionDigits: 0,
  }).format(value);
}

function calculateDCF(eps, growthRate, discountRate, sharesOutstanding) {
  const years = 5;
  const growth = growthRate / 100;
  const discount = discountRate / 100;
  let cashFlow = eps * 1000;
  let presentValue = 0;

  for (let i = 1; i <= years; i += 1) {
    cashFlow *= 1 + growth;
    presentValue += cashFlow / Math.pow(1 + discount, i);
  }

  const terminalValue = cashFlow * 15 / (discount - growth);
  const terminalDiscounted = terminalValue / Math.pow(1 + discount, years);
  const totalValue = presentValue + terminalDiscounted;
  return totalValue / (sharesOutstanding * 1_000_000);
}

function evaluate(value, current) {
  if (value <= 0 || current <= 0) return 'Không đủ dữ liệu';
  const ratio = value / current;
  if (ratio >= 1.2) return 'Hấp dẫn - có thể mua';
  if (ratio >= 0.9) return 'Cân nhắc đầu tư';
  return 'Quá cao - nên thận trọng';
}

calculateBtn.addEventListener('click', () => {
  const eps = Number(epsInput.value);
  const growthRate = Number(growthRateInput.value);
  const targetPE = Number(targetPEInput.value);
  const discountRate = Number(discountRateInput.value);
  const sharesOutstanding = Number(sharesOutstandingInput.value);
  const current = Number(currentPrice.value);

  if (!eps || !targetPE || !discountRate || !sharesOutstanding || !current) {
    alert('Vui lòng điền đầy đủ các giá trị số hợp lệ.');
    return;
  }

  const peModel = eps * targetPE;
  const dcfModel = calculateDCF(eps, growthRate, discountRate, sharesOutstanding);
  const fair = (peModel + dcfModel) / 2;

  peValue.textContent = formatCurrency(peModel);
  dcfValue.textContent = formatCurrency(dcfModel);
  fairValue.textContent = formatCurrency(fair);
  valuationLabel.textContent = evaluate(fair, current);
});
