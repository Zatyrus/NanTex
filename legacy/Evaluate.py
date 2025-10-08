import numpy as np


from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, rase, sam, msssim, vifp


def create_entry(desc, value):
    return {desc: value}


def evaluate_stack(feature_stack, pred_stack):
    l = [[[] for _ in range(10)] for _ in range(feature_stack.shape[1])]
    for i in range(feature_stack.shape[0]):
        for j in range(feature_stack.shape[1]):
            l[j][0].append(mse(pred_stack[i, j, ...], feature_stack[i, j, ...]))
            l[j][1].append(rmse(pred_stack[i, j, ...], feature_stack[i, j, ...]))
            l[j][2].append(psnr(pred_stack[i, j, ...], feature_stack[i, j, ...]))
            l[j][3].append(ssim(pred_stack[i, j, ...], feature_stack[i, j, ...]))
            l[j][4].append(msssim(pred_stack[i, j, ...], feature_stack[i, j, ...]))
            l[j][5].append(uqi(pred_stack[i, j, ...], feature_stack[i, j, ...]))
            l[j][6].append(ergas(pred_stack[i, j, ...], feature_stack[i, j, ...]))
            l[j][7].append(rase(pred_stack[i, j, ...], feature_stack[i, j, ...]))
            l[j][8].append(sam(pred_stack[i, j, ...], feature_stack[i, j, ...]))
            l[j][9].append(vifp(pred_stack[i, j, ...], feature_stack[i, j, ...]))

    eval_dict = {}
    for i in range(feature_stack.shape[1]):
        for j, eval in enumerate(
            [
                "MSE",
                "RMSE",
                "PSNR",
                "SSIM",
                "MSSSIM",
                "UQI",
                "ERGAS",
                "RASE",
                "SAM",
                "VIF",
            ]
        ):
            eval_dict[f"feature_{i}"][eval] = np.mean(l[i][j])


def stack_images(raw, pred, stack_raw=[], stack_pred=[]):
    return stack_raw.append(raw), stack_pred.append(pred)


def eval_image(img, pred, rescale: bool = True):
    if rescale:
        img = img / np.max(img) * 255
        img = img.astype(np.uint8)

        pred = pred / np.max(pred) * 255
        pred = pred.astype(np.float32)

    return {
        #'MSE':mse(img,pred),
        #'RMSE':rmse(img,pred),
        #'PSNR':psnr(img,pred),
        "SSIM": ssim(img, pred),
        "MSSSIM": msssim(img, pred),
        #'UQI':uqi(img,pred),
        #'ERGAS':ergas(img,pred),
        #'RASE':rase(img,pred),
        #'SAM':sam(img,pred),
        #'VIF':vifp(img,pred)
    }
