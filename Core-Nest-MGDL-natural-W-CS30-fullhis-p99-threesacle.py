from tqdm import tqdm
from time import time
import torch.nn as nn
import scipy.io as sio
from argparse import ArgumentParser
from skimage.measure import compare_ssim as ssim
from torch.utils.data._utils.collate import default_collate as collate_fn
from torch.utils.data import DataLoader
from pytorch_msssim import ssim as ssim_loss
from utils_together import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.cuda.amp import autocast
from torch.utils.data import BatchSampler, SequentialSampler

parser = ArgumentParser(description='Nest-MGDL')
parser.add_argument('--model_name', type=str, default='Nest-MGDL-fullhis-res-p99-ssim-threescale', help='model name')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=10, help='20，phase number of ISTA-Net-plus')
parser.add_argument('--growth-rate', type=int, default=32, help='G,32')
parser.add_argument('--num-layers', type=int, default=6, help='C,8，6')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cs_ratio', type=int, default=30, help='from {10, 25, 30, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--data_dir', type=str, default='cs_train400_png', help='training data directory')
parser.add_argument('--rgb_range', type=int, default=1, help='value range 1 or 255')
parser.add_argument('--n_channels', type=int, default=1, help='1 for gray, 3 for color')
parser.add_argument('--patch_size', type=int, default=33*3, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir_org', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--ext', type=str, default='.png', help='training data directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set,Set11/SetBSD68/Urban100HR')
parser.add_argument('--test_cycle', type=int, default=20, help='epoch number of each test cycle')
parser.add_argument('--overlap_size', type=str, default=16, help='overlap pixel: 8')
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--mode', type=str, default='test', help='train or test')
args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list

test_name = args.test_name
test_cycle = args.test_cycle

test_dir = os.path.join(args.data_dir_org, test_name)
filepaths = glob.glob(test_dir + '/*.*')

ImgNum = len(filepaths)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

ratio_dict = {1: 10, 4: 43, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]
n_output = 1089
batch_size = 8

# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, args.cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']

def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)

###########################################################################
class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        # Branch
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.tail_cat = nn.Sequential(nn.Conv2d(in_channels + 2*out_channels, out_channels, kernel_size=1),nn.ReLU(inplace=True))

    def forward(self, x):
        x_feat = torch.cat([self.branch1(x),self.branch2(x),self.branch3(x)], dim=1)
        return torch.cat([x, self.tail_cat(x_feat)], 1)
# ###########################################################################
class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, 1, kernel_size=1) # output 1 channel

    def forward(self, x):
        return x[:,0:1,:,:] +self.lff(self.layers(x))  # local residual learning
###########################################################################
class BasicBlock(torch.nn.Module):
    def __init__(self,growth_rate, num_layers, in_channels,max_history):
        super(BasicBlock, self).__init__()
        self.G = growth_rate
        self.C = num_layers

        self.max_history = max_history  # 控制历史信息长度
        self.rdb = RDB(in_channels, self.G, self.C)  # local residual learning

    def forward(self, yold, z, PhiWeight, PhiTWeight, PhiTb,history):
        # 梯度更新（确保 lambda_step 定义正确）
        x = - PhiTPhi_fun(z, PhiWeight, PhiTWeight) +  PhiTb

        # 构建输入通道，包括当前 x 和历史
        if history.numel() == 0:
            x_input_cat = x
        else:
            # 更新历史信息
            x_input_cat = torch.cat([x, history], dim=1)
            if x_input_cat.shape[1] > self.max_history:
                x_input_cat = x_input_cat[:, : self.max_history, :, :]

        # RDB处理
        y_tmp = torch.cat([yold, x_input_cat], dim=1)
        ynew = self.rdb(y_tmp)

        return [ynew,x_input_cat]
##########################################################################
# DefineMGD
class MGD(nn.Module):
    def __init__(self, LayerNo,growth_rate, num_layers):
        super(MGD, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.Sp = nn.Softplus()

        # Learnable parameters
        self.w_gamma = nn.Parameter(torch.Tensor([0.5]))
        self.b_gamma = nn.Parameter(torch.Tensor([0]))

        self.G = growth_rate
        self.C = num_layers

        self.max_history = LayerNo  # 控制历史信息长度

        for i in range(LayerNo):
            if i > self.max_history-1:
                in_channels = self.max_history + 1
            else:
                in_channels = i + 1 + 1
            onelayer.append(BasicBlock(self.G, self.C,in_channels,self.max_history))  # share feature extrator

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, x, Phi, n_input):
        batchx = x
        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
        Phix = F.conv2d(batchx, PhiWeight, padding=0, stride=33, bias=None)

        PhiTWeight = Phi.t().contiguous().view(n_output, n_input, 1, 1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)

        xold = PhiTb
        yold = xold
        history = torch.tensor([]).to(xold.device)

        for i in range(self.LayerNo):
            gamma_ = self.Sp(self.w_gamma * i + self.b_gamma)
            x_coef = torch.tensor([1 - gamma_, gamma_], device=x.device)
            gamma_s = F.softmax(x_coef, dim=0)
            z = gamma_s[0] * xold + gamma_s[1] * yold
            [ynew,history] = self.fcs[i](yold, z, PhiWeight, PhiTWeight, PhiTb, history)
            # Nesterov更新
            xnew = gamma_s[0] * xold + gamma_s[1] * ynew
            xold = xnew
            yold = ynew

        x_final = xnew
        return [x_final]
###########################################################################
model = MGD(layer_num, args.growth_rate, args.num_layers)
model = nn.DataParallel(model)
model = model.to(device)

print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len
########################################################################
def get_val_result(model,mode,save_image_str, Phi, test_name, data_dir_org, overlap_size):
    model.eval()
    with torch.no_grad():
        test_set_path = os.path.join(data_dir_org, test_name)
        test_set_path = glob.glob(test_set_path + '/*.*')
        ImgNum = len(test_set_path)
        PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
        SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
        step = overlap_size

        if not os.path.exists(save_image_str):
            os.makedirs(save_image_str)

        time_all = 0
        for img_no in range(ImgNum):
            imgName = test_set_path[img_no]
            start = time()
            start1 = time()
            [Iorg, row, col, Ipad, pad_top, pad_left, Img_rec_yuv] = imread_CS_py_paddu(imgName, device, args.patch_size)  # padding分在四周
            end1 = time()
            time1 = (end1 - start1)
            start2 = time()
            patches = img2patches_torch(Ipad, (args.patch_size, args.patch_size), (step, step))
            patches_batch = patches / 255.0
            inputs = patches_batch
            end2 = time()
            time2 = (end2 - start2)
            start3 = time()
            output = torch.zeros_like(inputs, dtype=torch.float16).cuda()  # 假设 inputs 已经是 torch.Tensor

            # 准备 batch 索引列表
            batch_list = list(BatchSampler(SequentialSampler(output), batch_size=args.eval_batch_size, drop_last=False))
            input_batches = list(BatchSampler(inputs, batch_size=args.eval_batch_size, drop_last=False))

            def model_infer(idx, list_data):
                batch_x = collate_fn(list_data).cuda(non_blocking=True)
                with torch.no_grad():
                    with autocast():
                        batch_y = model(batch_x, Phi, n_input)
                if isinstance(batch_y, list):
                    batch_y = batch_y[0]
                return idx, batch_y

            # 使用线程池并发推理
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(model_infer, idx, list_data)
                           for idx, list_data in enumerate(input_batches)]

                for future in as_completed(futures):
                    idx, batch_y = future.result()
                    list_tmp = batch_list[idx]
                    output[list_tmp, :, :, :] = batch_y.to(output.dtype)

            end3 = time()
            time3 = (end3 - start3)
            start4 = time()

            output = unpatch2d_torch(output, Ipad.shape, (step, step)).squeeze()
            images_recovered = output[pad_top:pad_top + row, pad_left:pad_left + col].cpu().numpy()

            isnan = np.isnan(images_recovered)  # 判断是否存在边缘重建出错
            if True in isnan:
                images_recovered[np.isnan(images_recovered)] = np.nanmean(images_recovered)

            isnan = np.isnan(images_recovered)  # 判断是否存在边缘重建出错
            if True in isnan:
                images_recovered[np.isnan(images_recovered)] = np.nanmean(images_recovered)

            end4 = time()
            time4 = (end4 - start4)

            print('Part Time', time1, time2, time3, time4)

            end = time()
            time_all = time_all + (end - start)

            X_rec = np.clip(images_recovered, 0, 1).astype(np.float64)

            rec_PSNR = psnr(X_rec * 255, Iorg.astype(np.float64))
            rec_SSIM = ssim(X_rec * 255, Iorg.astype(np.float64), data_range=255)
            print('psnr=%.5f' % rec_PSNR)
            # save image
            if mode == 'test':
                print("[%02d/%02d] Run time for %s is %.5f, PSNR is %.3f, SSIM is %.5f" % (
                    img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM))
                Img_rec_yuv[:, :, 0] = X_rec * 255

                im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
                im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

                name_str = imgName.split("/")
                name_str2 = name_str[2].split(".")
                cv2.imwrite("./%s/%s_PSNR_%.3f_SSIM_%.5f.png" % (
                    save_image_str, name_str2[0], rec_PSNR, rec_SSIM), im_rec_rgb)

            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM

        mean_time = time_all / ImgNum
    return PSNR_All,SSIM_All, mean_time
##########################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model_dir = "./%s/%s_layer_%d_ratio_%d_lr_%.4f" % (args.model_dir, args.model_name, layer_num, args.cs_ratio, learning_rate)
log_file_name = "./%s/%s_Log_layer_%d_ratio_%d_lr_%.4f.txt" % (args.log_dir, args.model_name, layer_num, args.cs_ratio, learning_rate)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if start_epoch > 0:
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, start_epoch)))

Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.to(device)

if args.mode == 'train':
    training_data = SlowDataset(args)
    rand_loader = DataLoader(dataset=training_data, batch_size=batch_size, num_workers=8,
                             shuffle=True)
    # Training loop
    for epoch_i in range(start_epoch + 1, end_epoch + 1):
        # tqdm 包裹数据加载器，显示批次训练进度
        loop = tqdm(rand_loader, desc=f"Epoch [{epoch_i}/{end_epoch}]", leave=False)
        for data in loop:

            batch_x = data.view(-1, 1, args.patch_size, args.patch_size)
            batch_x = batch_x.to(device)

            [x_output] = model(batch_x, Phi,n_input)

            # Compute and print loss
            loss_discrepancy = torch.mean(torch.abs(x_output - batch_x))
            loss_ssim = 1 - ssim_loss(x_output, batch_x, data_range=1.0, size_average=True)

            loss_all = loss_discrepancy + loss_ssim  #+ 0.01 * loss_constraint

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            loop.set_postfix(loss=loss_all.item())  # tqdm中显示当前 loss

        output_loss = str(datetime.now()) + " [%d/%d] Total loss: %.4f, L1 loss: %.4f, SSIM loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item(),loss_ssim.item())
        print(output_loss)

        PSNR_All, SSIM_All, mean_time = get_val_result(model, args.mode, model_dir, Phi, test_name,
                                                       args.data_dir_org, args.overlap_size)
        output_data = "CS ratio is %d, Avg time is %.5f, Avg Proposed PSNR/SSIM is %.3f/%.5f, Epoch number of model is %d \n" % (
            args.cs_ratio, mean_time, np.mean(PSNR_All), np.mean(SSIM_All), epoch_i)
        print(output_data)

        # save result
        output_data = [epoch_i, np.mean(PSNR_All), np.std(PSNR_All), np.mean(SSIM_All), np.std(SSIM_All)]
        output_file = open(model_dir + "/log_PSNR.txt", 'a')
        for fp in output_data:  # write data in txt
            output_file.write(str(fp))
            output_file.write(',')
        output_file.write('\n')  # line feed
        output_file.close()

        if epoch_i % test_cycle == 0:
            print('model saved!')
            torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters

elif args.mode == 'test':
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, args.end_epoch)))
    model_dir_result = "./%s/CS_%s_ratio_%d" % (args.result_dir, args.model_name, args.cs_ratio)
    print('Test')
    PSNR_All, SSIM_All, mean_time = get_val_result(model, args.mode, model_dir_result, Phi,
                                                   test_name, args.data_dir_org, args.overlap_size)
    output_data = "CS ratio is %d, Avg time is %.5f, Avg Proposed PSNR/SSIM is %.3f/%.5f, Epoch number of model is %d \n" % (
        args.cs_ratio, mean_time, np.mean(PSNR_All), np.mean(SSIM_All), args.end_epoch)
    print(output_data)


