import torch
from utils import clamp
from generate_wm import generate_watermark,generate_watermark_ori
import torch.nn.functional as F

class BVMR_WV(object):
    def __init__(self,model,args,epsilon,start_epsilon,step_alpha,upper_limit,lower_limit):
        self.model = model
        self.epsilon = epsilon
        self.start_epsilon = start_epsilon
        self.step_alpha = step_alpha
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.args = args

    def Clean(self,img_source,logo_path,seed):
        wm, mask = generate_watermark_ori(img_source, logo_path, seed)
        wm = wm * 255
        wm = torch.unsqueeze(wm.cuda(), 0)
        wm = (wm / 127.5 - 1)
        output = self.model(wm)
        guess_images, guess_mask = output[0], output[1]
        expanded_guess_mask = guess_mask.repeat(1, 3, 1, 1)
        reconstructed_pixels = guess_images * expanded_guess_mask
        reconstructed_images = wm * (1 - expanded_guess_mask) + reconstructed_pixels
        transformed_guess_mask = expanded_guess_mask * 2 - 1
        return wm/2+0.5,reconstructed_images/2+0.5,transformed_guess_mask/2+0.5

    def RN(self,img_source,logo_path,seed):
        random_noise = torch.zeros_like(img_source).cuda()
        for i in range(len(self.epsilon)):
            random_noise[:, i, :, :].uniform_(-self.start_epsilon[i][0][0].item(), self.start_epsilon[i][0][0].item())
        random = clamp(img_source  + random_noise, self.lower_limit, self.upper_limit)
        random, _ = generate_watermark_ori(random, logo_path, seed)
        random = random * 255
        random = torch.unsqueeze(random.cuda(), 0)
        random = (random / 127.5 - 1)
        r_output = self.model(random)

        r_guess_images, r_guess_mask = r_output[0], r_output[1]
        r_expanded_guess_mask = r_guess_mask.repeat(1, 3, 1, 1)
        r_reconstructed_pixels = r_guess_images * r_expanded_guess_mask
        r_reconstructed_images = random * (1 - r_expanded_guess_mask) + r_reconstructed_pixels
        r_transformed_guess_mask = r_expanded_guess_mask * 2 - 1
        return random/2+0.5,r_reconstructed_images/2+0.5, r_transformed_guess_mask/2+0.5

    def DWV(self,img_source,logo_path,seed):
        img = img_source*255
        delta1 = torch.zeros_like(img).cuda()
        delta1.requires_grad = True
        for i in range(50):
            mid_adv1 = (img + delta1) / 127.5 - 1
            mid_output = self.model(mid_adv1)
            mid_guess_images, mid_guess_mask = mid_output[0], mid_output[1]
            mid_expanded_guess_mask = mid_guess_mask.repeat(1, 3, 1, 1)
            mid_reconstructed_pixels = mid_guess_images * mid_expanded_guess_mask
            mid_reconstructed_images = mid_adv1 * (1 - mid_expanded_guess_mask) + mid_reconstructed_pixels
            mid_reconstructed_images = mid_reconstructed_images / 2 + 0.5
            loss = F.mse_loss((img / 255).data, mid_reconstructed_images.float())
            loss.backward()
            grad = delta1.grad.detach()
            d = delta1
            d = clamp(d + 255* self.step_alpha * torch.sign(grad), -self.epsilon*255, self.epsilon*255)
            delta1.data = d
            delta1.grad.zero_()

        adv1 = img + delta1
        adv1 = clamp(adv1, self.lower_limit*255 , self.upper_limit*255 )
        adv1, _ = generate_watermark_ori(adv1 / 255, logo_path, seed)
        adv1 = torch.unsqueeze(adv1.cuda(), 0) * 255
        adv1 = (adv1 / 127.5 - 1)
        adv1_output = self.model(adv1)
        adv1_guess_images, adv1_guess_mask = adv1_output[0], adv1_output[1]
        adv1_expanded_guess_mask = adv1_guess_mask.repeat(1, 3, 1, 1)
        adv1_reconstructed_pixels = adv1_guess_images * adv1_expanded_guess_mask
        adv1_reconstructed_images = adv1 * (1 - adv1_expanded_guess_mask) + adv1_reconstructed_pixels
        adv1_transformed_guess_mask = adv1_expanded_guess_mask * 2 - 1

        return adv1/2+0.5,adv1_reconstructed_images/2+0.5, adv1_transformed_guess_mask/2+0.5



    def IWV(self,img_source,logo_path,seed):
        img = img_source *255
        delat2 = torch.zeros_like(img).cuda()
        delat2.requires_grad = True
        for i in range(50):
            mid_adv2 = (img + delat2) / 127.5 - 1
            mid_output = self.model(mid_adv2)
            mid_guess_images, mid_guess_mask = mid_output[0], mid_output[1]
            mid_expanded_guess_mask = mid_guess_mask.repeat(1, 3, 1, 1)
            mid_reconstructed_pixels = mid_guess_images * mid_expanded_guess_mask
            mid_reconstructed_images = mid_adv2 * (1 - mid_expanded_guess_mask) + mid_reconstructed_pixels
            mid_reconstructed_images = mid_reconstructed_images / 2 + 0.5
            mid_transformed_guess_mask = mid_expanded_guess_mask * 2 - 1
            mid_transformed_guess_mask = mid_transformed_guess_mask / 2 + 0.5
            mask_black = torch.zeros_like(mid_transformed_guess_mask).cuda()
            loss = F.mse_loss((img / 255).data, mid_reconstructed_images.float()) + 2*F.mse_loss((mask_black).data,
                                                                                                     mid_transformed_guess_mask.float())
            loss.backward()
            grad = -delat2.grad.detach()
            d = delat2
            d = clamp(d + 255 * self.step_alpha * torch.sign(grad), -self.epsilon * 255, self.epsilon * 255)
            delat2.data = d
            delat2.grad.zero_()

        adv2 = img + delat2
        adv2 = clamp(adv2, self.lower_limit*255 , self.upper_limit*255)
        adv2, _ = generate_watermark_ori(adv2 / 255, logo_path, seed)
        adv2 = torch.unsqueeze(adv2.cuda(), 0) * 255
        adv2 = (adv2 / 127.5 - 1)
        adv2_output = self.model(adv2)
        adv2_guess_images, adv2_guess_mask = adv2_output[0], adv2_output[1]
        adv2_expanded_guess_mask = adv2_guess_mask.repeat(1, 3, 1, 1)
        adv2_reconstructed_pixels = adv2_guess_images * adv2_expanded_guess_mask
        adv2_reconstructed_images = adv2 * (1 - adv2_expanded_guess_mask) + adv2_reconstructed_pixels
        adv2_transformed_guess_mask = adv2_expanded_guess_mask * 2 - 1
        return adv2 / 2 + 0.5, adv2_reconstructed_images / 2 + 0.5, adv2_transformed_guess_mask / 2 + 0.5