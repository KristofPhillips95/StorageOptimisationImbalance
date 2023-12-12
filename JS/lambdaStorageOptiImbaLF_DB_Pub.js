import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import {
  DynamoDBDocumentClient,
  ScanCommand,
  PutCommand,
  GetCommand,
  DeleteCommand,
} from "@aws-sdk/lib-dynamodb";

const client = new DynamoDBClient({});

const dynamo = DynamoDBDocumentClient.from(client);

const tableName = "StorageOptimPublicOut";

export const handler = async (event, context) => {
  let body;
  let statusCode = 200;
  const headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
  };

  try {
    switch (event.routeKey) {
      case "DELETE /items/{id}":
        await dynamo.send(
          new DeleteCommand({
            TableName: tableName,
            Key: {
              id: event.pathParameters.id,
            },
          })
        );
        body = `Deleted item ${event.pathParameters.id}`;
        break;
        case "DELETE /allitems":
            const allItemsToDelete = await dynamo.send(new ScanCommand({ TableName: tableName }));
            const itemsToDelete = allItemsToDelete.Items;
            for (const item of itemsToDelete) {
              await dynamo.send(new DeleteCommand({
                TableName: tableName,
                Key: {
                  id: item.id // Ensure 'id' is the correct primary key or identifier
                }
              }));
            }
            body = `Deleted all items`;
            break;
      case "GET /items/{id}":
        body = await dynamo.send(
          new GetCommand({
            TableName: tableName,
            Key: {
              id: event.pathParameters.id,
            },
          })
        );
        body = body.Item;
        break;
      case "GET /items":
        body = await dynamo.send(
          new ScanCommand({ TableName: tableName })
        );
        body = body.Items;
        break;
        case "PUT /items":
            let requestJSON = JSON.parse(event.body);
            
            // Write the new item to the database
            await dynamo.send(
              new PutCommand({
                TableName: tableName,
                Item: {
                  id: requestJSON.id,
                  time: requestJSON.time,
                  imba_price: requestJSON.imba_price,
                  imba_prce_fc: requestJSON.imba_price_fc,
                  charge: requestJSON.charge,
                  soc: requestJSON.soc,
                  fc_spread: requestJSON.fc_spread
                },
              })
            );
          
            // Logic to keep only the latest 24 items
            const allItems = await dynamo.send(new ScanCommand({ TableName: tableName }));
            const items = allItems.Items;
          
            if (items.length > 24) {
              // Sort the items by timestamp or an attribute representing their order
              const sortedItems = items.sort((a, b) => b.id - a.id);

          
              // Delete surplus items beyond the latest 24
              const itemsToDelete = sortedItems.slice(24);
              for (const item of itemsToDelete) {
                await dynamo.send(new DeleteCommand({
                  TableName: tableName,
                  Key: {
                    id: item.id // Replace with your primary key or identifier
                  }
                }));
              }
            }
          
            body = `Put item ${requestJSON.id}`;
            break;
      default:
        throw new Error(`Unsupported route: "${event.routeKey}"`);
    }
  } catch (err) {
    statusCode = 400;
    body = err.message;
  } finally {
    body = JSON.stringify(body);
  }

  return {
    statusCode,
    body,
    headers,
  };
};