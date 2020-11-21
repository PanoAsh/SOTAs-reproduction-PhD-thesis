"""

x = self.resnet.conv1(x)
x = self.resnet.bn1(x)
x = self.resnet.relu(x)
x = self.resnet.maxpool(x)

x_depth = self.resnet_depth.conv1(x_depth)
x_depth = self.resnet_depth.bn1(x_depth)
x_depth = self.resnet_depth.relu(x_depth)
x_depth = self.resnet_depth.maxpool(x_depth)

x_fss = self.resnet_fs.conv1(x_fss)
x_fss = self.resnet_fs.bn1(x_fss)
x_fss = self.resnet_fs.relu(x_fss)
x_fss = self.resnet_fs.maxpool(x_fss)

# layer0 merge
# temp_fss = x_fss.unsqueeze(0)
# temp_fss = temp_fss.permute(0, 2, 1, 3, 4)
# temp_fss = self.TDConv_1(temp_fss)
# temp_fss = temp_fss.squeeze(2)
temp = x_depth.mul(self.atten_depth_channel_0(x_depth))
temp = temp.mul(self.atten_depth_spatial_0(temp))
# x = x + temp + temp_fss
x = x + temp
# layer0 merge end

x1 = self.resnet.layer1(x)
x1_depth = self.resnet_depth.layer1(x_depth)
# x1_fss = self.resnet_fs.layer1(x_fss)

# layer1 merge
# temp_fss = x1_fss.unsqueeze(0)
# temp_fss = temp_fss.permute(0, 2, 1, 3, 4)
# temp_fss = self.TDConv_2(temp_fss)
# temp_fss = temp_fss.squeeze(2)
temp = x1_depth.mul(self.atten_depth_channel_1(x1_depth))
temp = temp.mul(self.atten_depth_spatial_1(temp))
# x1 = x1 + temp + temp_fss
x1 = x1 + temp
# layer1 merge end

x2 = self.resnet.layer2(x1)
x2_depth = self.resnet_depth.layer2(x1_depth)
# x2_fss = self.resnet_fs.layer2(x1_fss)

# layer2 merge
# temp_fss = x2_fss.unsqueeze(0)
# temp_fss = temp_fss.permute(0, 2, 1, 3, 4)
# temp_fss = self.TDConv_3(temp_fss)
# temp_fss = temp_fss.squeeze(2)
temp = x2_depth.mul(self.atten_depth_channel_2(x2_depth))
temp = temp.mul(self.atten_depth_spatial_2(temp))
# x2 = x2 + temp + temp_fss
x2 = x2 + temp
# layer2 merge end

x2_1 = x2

x3_1 = self.resnet.layer3_1(x2_1)
x3_1_depth = self.resnet_depth.layer3_1(x2_depth)
# x3_1_fss = self.resnet_fs.layer3_1(x2_fss)

# layer3_1 merge
# temp_fss = x3_1_fss.unsqueeze(0)
# temp_fss = temp_fss.permute(0, 2, 1, 3, 4)
# temp_fss = self.TDConv_4(temp_fss)
# temp_fss = temp_fss.squeeze(2)
temp = x3_1_depth.mul(self.atten_depth_channel_3_1(x3_1_depth))
temp = temp.mul(self.atten_depth_spatial_3_1(temp))
# x3_1 = x3_1 + temp + temp_fss
x3_1 = x3_1 + temp
# layer3_1 merge end

x4_1 = self.resnet.layer4_1(x3_1)
x4_1_depth = self.resnet_depth.layer4_1(x3_1_depth)
# x4_1_fss = self.resnet_fs.layer4_1(x3_1_fss)

# layer4_1 merge
# temp_fss = x4_1_fss.unsqueeze(0)
# temp_fss = temp_fss.permute(0, 2, 1, 3, 4)
# temp_fss = self.TDConv_5(temp_fss)
# temp_fss = temp_fss.squeeze(2)
temp = x4_1_depth.mul(self.atten_depth_channel_4_1(x4_1_depth))
temp = temp.mul(self.atten_depth_spatial_4_1(temp))
# x4_1 = x4_1 + temp + temp_fss
x4_1 = x4_1 + temp
# layer4_1 merge end

# produce initial saliency map by decoder1
x2_1 = self.rfb2_1(x2_1)
x3_1 = self.rfb3_1(x3_1)
x4_1 = self.rfb4_1(x4_1)
attention_map = self.agg1(x4_1, x3_1, x2_1)

# Refine low-layer features by initial map
x, x1, x5 = self.HA(attention_map.sigmoid(), x, x1, x2)

# produce final saliency map by decoder2
x0_2 = self.rfb0_2(x)
x1_2 = self.rfb1_2(x1)
x5_2 = self.rfb5_2(x5)
y = self.agg2(x5_2, x1_2, x0_2)  # *4

# PTM module
y = self.agant1(y)
y = self.deconv1(y)
y = self.agant2(y)
y = self.deconv2(y)
y = self.out2_conv(y)

return self.upsample(attention_map), y


#self.TDConv_1 = nn.Conv3d(64, 64, (12, 1, 1))
        #self.TDConv_2 = nn.Conv3d(256, 256, (12, 1, 1))
        #self.TDConv_3 = nn.Conv3d(512, 512, (12, 1, 1))
        #self.TDConv_4 = nn.Conv3d(1024, 1024, (12, 1, 1))
        #self.TDConv_5 = nn.Conv3d(2048, 2048, (12, 1, 1))

self.convLSTM = ConvLSTM(input_channels=64, hidden_channels=[64, 32, 64],
                 kernel_size=5, step=4, effective_step=[2, 4, 8])

for i in range(12):
            cv2.imwrite(str(i) + '.png', x_fss[i].permute(1, 2, 0).cpu().data.numpy() * 255)
        cv2.imwrite('img.png', x.squeeze(0).permute(1, 2, 0).cpu().data.numpy() * 255)
        cv2.imwrite('depth.png', x_depth.squeeze(0).permute(1, 2, 0).cpu().data.numpy() * 255)

          for idx in range(1):
            temp_fss = x_fss[idx].mul(self.atten_fs_channel_0(x_fss[idx]))
            temp_fss = temp_fss.mul(self.atten_fs_spatial_0(temp_fss))
            x = x + temp_fss

  #cv2.imwrite('a.png', self.upsample(attention_map).squeeze(0).permute(1, 2, 0).cpu().data.numpy() * 255)
        #cv2.imwrite('y.png', y.squeeze(0).permute(1, 2, 0).cpu().data.numpy() * 255)
        #cv2.imwrite('fs.png', attention_map_fusion.squeeze(0).permute(1, 2, 0).cpu().data.numpy() * 255)

  # late fusion of refocus and RGB+depth
        attention_map_fusion = torch.cat((attention_map, attention_map_fss), 0)
        attention_map_fusion = attention_map_fusion.permute(1, 0, 2, 3)
        attention_map_fusion = self.FS_fusion(self.upsample(attention_map_fusion))
        attention_map_fusion_resize = self.downsample4(attention_map_fusion)

        # Refine low-layer features by initial map
        x, x1, x5 = self.HA(attention_map_fusion_resize.sigmoid(), x, x1, x2)

            for i in range(12):
            cv2.imwrite(str(i) + '.png', x_fss[:,i,:,:].permute(1, 2, 0).cpu().data.numpy() * 255)
        cv2.imwrite('img.png', x.squeeze(0).permute(1, 2, 0).cpu().data.numpy() * 255)
        cv2.imwrite('depth.png', x_depth.squeeze(0).permute(1, 2, 0).cpu().data.numpy() * 255)

            #img = img.filter(ImageFilter.FIND_EDGES)
                #img = ImageOps.grayscale(img)

     img = np.asarray(img)
                img_seg = segmentation.slic(img, compactness=10, n_segments=100)
                super_pix = color.label2rgb(img_seg, img, kind='avg')
                img = Image.fromarray(super_pix)

       attention_map_fss = self.FS_Feat(attention_map_fss)
        attention_map_fss = attention_map_fss.unsqueeze(0).permute(0, 2, 1, 3, 4)
        attention_map_fss = self.FS_TS_Feat(attention_map_fss)
        attention_map_fss = self.FS_bn(attention_map_fss)
        attention_map_fss = self.FS_relu(attention_map_fss)
        attention_map_fss = attention_map_fss.squeeze(0).permute(1, 0, 2, 3)

"""