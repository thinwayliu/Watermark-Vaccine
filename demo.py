import  argparse
from utils import *
from model.WDNet import *
from model.BVMR import *
from model.SplitNet import *
from tqdm import tqdm

def main():

    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)
    mu = torch.tensor(mean).view(3, 1, 1).cuda()
    std = torch.tensor(std).view(3, 1, 1).cuda()

    upper_limit = ((1 - mu) / std)
    lower_limit = ((0 - mu) / std)

    epsilon = (args.epsilon / 255.) / std
    start_epsilon = (args.start_epsilon / 255.) / std
    step_alpha = (args.step_alpha / 255.) / std

    np.random.seed(args.seed)

    if args.model == 'WDNet':
        model = generator(3, 3)
        model.eval()
        optimizer = WDNet_WV(model,args,epsilon,start_epsilon,step_alpha,upper_limit,lower_limit)
        model.load_state_dict(torch.load(os.path.join(args.load_path), map_location='cpu'))
        model.cuda()
    elif args.model == 'BVMR':
        _opt = load_globals('./', {}, override=False)
        _device = torch.device('cuda:0')
        model = init_nets(_opt, './', _device, 11).eval()
        optimizer = BVMR_WV(model,args,epsilon,start_epsilon,step_alpha,upper_limit,lower_limit)
    elif args.model == 'SplitNet':
        model = models.__dict__['vvv4n']().cuda()
        model.load_state_dict(torch.load(args.load_path)['state_dict'])
        model.eval()
        optimizer = SplitNet_WV(model, args, epsilon, start_epsilon, step_alpha, upper_limit, lower_limit)

    # else:
    #     EOFError



    transform_norm = transforms.Compose([transforms.ToTensor()])



    for t in tqdm(range(args.num_img)):
        t += 1
        logo_index = np.random.randint(1, 161)
        imageJ_path = os.path.join(args.image_path ,'%s.jpg' % (t))
        logo_path = os.path.join(args.logo_path ,'%s.png' % (logo_index))

        img_J = Image.open(imageJ_path)
        img_source = transform_norm(img_J)
        img_source = torch.unsqueeze(img_source.cuda(), 0)


        # 初始化扰动
        seed = np.random.randint(0, 1000)

        # Clean output
        wm, clean_pred, clean_mask = optimizer.Clean(img_source,logo_path,seed)
        clean_show = img_show(wm, clean_pred, clean_mask,  config='Clean')


        # Random Noise
        random, random_pred, random_mask = optimizer.RN(img_source,logo_path,seed)
        random_show = img_show(random, random_pred, random_mask,  config='RN')


        # Disrupting Watermark Vaccine
        adv1, adv1_pred, adv1_mask = optimizer.DWV(img_source,logo_path,seed)
        adv1_show = img_show(adv1, adv1_pred, adv1_mask,  config='DWV')

        # Inerasable Wateramark Vaccine
        adv2, adv2_pred, adv2_mask = optimizer.IWV(img_source,logo_path,seed)
        adv2_show = img_show(adv2, adv2_pred, adv2_mask,  config='IWV')

        all_images = [clean_show,random_show,adv1_show,adv2_show]
        all_images_show = all_image_concat(all_images)
        all_images_show.save(os.path.join(args.save_path,'effect_show-%s-%d.jpg'%(args.model,t)))





if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, help='Watermark Removal Network (WDNet,SplitNet,BVMR)', default='WDNet')
    argparser.add_argument('--load_path', type=str,default='./WDNet_G.pkl')
    argparser.add_argument('--image_path', type=str,default='./dataset/CLWD/test/Watermark_free_image/')
    argparser.add_argument('--logo_path', type=str,default='./dataset/CLWD/watermark_logo/train_color')
    argparser.add_argument('--epsilon', type=int, help='the bound of perturbation', default=8)
    argparser.add_argument('--start_epsilon', type=int, help='the bound of random noise', default=8)
    argparser.add_argument('--step_alpha', type=int, help='step size', default=2)
    argparser.add_argument('--seed', type=int, help='random seed', default=160)
    argparser.add_argument('--num_img', type=int, help='imgsz', default=20)
    argparser.add_argument('--attack_iter', type=int, default=50)
    argparser.add_argument('--save_path', type=str,default='./visualization/')

    args = argparser.parse_args()

    main()
