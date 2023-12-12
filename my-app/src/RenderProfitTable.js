export function renderProfitTable(values){
    return (
      <table style={{ border: '1px solid black', padding: '10px' }}>
        <thead>
          <tr>
            <th>Imbalance price exposure (euro)</th>
            <th>Day</th>
            <th>Month</th>
            <th>Year</th>
          </tr>
        </thead>
        <tbody>
          {values.map((item, index) => (
            <tr key={index}>
              <td>{item.name}</td>
              <td>{item.day}</td>
              <td>{item.month}</td>
              <td>{item.year}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  };