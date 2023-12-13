// Helper function to get the start of the day
function getStartOfDay() {
  const now = new Date();
  return new Date(now.getFullYear(), now.getMonth(), now.getDate());
}

// Helper function to get the start of the week
function getStartOfWeek() {
  const now = new Date();
  const dayOfWeek = now.getDay();
  const diff = now.getDate() - dayOfWeek + (dayOfWeek === 0 ? -6 : 1);  // Adjust for Sunday
  return new Date(now.setDate(diff));
}

// Helper function to get the start of the month
function getStartOfMonth() {
  const now = new Date();
  return new Date(now.getFullYear(), now.getMonth(), 1);
}
function processLtsData(data_lts){
  console.log(getStartOfDay())
  let values = [
    { name: 'Charge cost', day: 100, month: 300,year: 1000 },
    { name: 'Discharge revenue', day: 150, month: 200,year: 800 },
    { name: 'Profit', day: 50, month: 100,year: 200 },
    // Add other specific numbers related to your data
  ];

  const todayCharge = summedExposureSince(data_lts, getStartOfDay(), 'charge');
  const thisWeekCharge = summedExposureSince(data_lts, getStartOfWeek(), 'charge');
  const thisMonthCharge = summedExposureSince(data_lts, getStartOfMonth() , 'charge');

  const todayDisCharge = summedExposureSince(data_lts, getStartOfDay(), 'discharge');
  const thisWeekDisCharge = summedExposureSince(data_lts, getStartOfWeek(), 'discharge');
  const thisMonthDisCharge = summedExposureSince(data_lts, getStartOfMonth() , 'discharge');


  values = [
    { name: 'Charge cost', day: todayCharge, week: thisWeekCharge,month: thisMonthCharge },
    { name: 'Discharge revenue', day: todayDisCharge, week: thisWeekDisCharge,month: thisMonthDisCharge },
    { name: 'Profit', day: todayDisCharge- todayCharge, week: thisWeekCharge-thisWeekDisCharge,month: thisMonthCharge-thisMonthDisCharge },
    // Add other specific numbers related to your data
  ];
  return values
}
function summedExposureSince(data_lts,startDate,exposureType){
  // Convert the given date to a JavaScript Date object
  const parsedStartDate = new Date(startDate);

  const data_since_start = data_lts
   .filter(item=> {
    const item_date = new Date(item.id)
    return item_date >= parsedStartDate
   } )
  const summedProduct = calculate_summed_product(data_since_start,exposureType)
  return summedProduct
}

function calculate_summed_product(data,exposureType){
  return data.reduce((sum,entry)=>{
    let product;
    if(exposureType =="charge"){
      product = entry.charge * entry.Price
    }
    if(exposureType =="discharge"){
      product = entry.discharge * entry.Price
    }
    return sum + product
  },0)
}

export function RenderProfitTable(values,data_lts) {
  // const values = FetchLtsData();
  const values_2 = processLtsData(data_lts)
  // console.log(data_lts)

  return (
    <table style={{ border: '1px solid black', padding: '10px' }}>
      <thead>
        <tr>
          <th>Imbalance price exposure (euro)</th>
          <th>Day</th>
          <th>Week</th>
          <th>Month</th>
        </tr>
      </thead>
      <tbody>
        {values_2.map((item, index) => (
          <tr key={index}>
            <td>{item.name}</td>
            <td>{item.day}</td>
            <td>{item.week}</td>
            <td>{item.month}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
