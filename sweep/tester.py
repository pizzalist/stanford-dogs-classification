# import torch

# def Test(model, test_DL, DEVICE):
#     model.eval()
#     with torch.no_grad():
#         correct_predictions = 0
#         total_samples = 0
#         for x_batch, y_batch in test_DL:
#             x_batch = x_batch.to(DEVICE)
#             y_batch = y_batch.to(DEVICE)
#             # inference
#             y_hat = model(x_batch)
#             # accuracy accumulation
#             pred = y_hat.argmax(dim=1)
#             correct_predictions += torch.sum(pred == y_batch).item()
#             total_samples += len(y_batch)
#         accuracy = correct_predictions / total_samples * 100
#     print(f"Test accuracy: {correct_predictions}/{total_samples} ({round(accuracy, 1)} %)")

# def count_params(model):
#     num = sum([p.numel() for p in model.parameters() if p.requires_grad])
#     return num

