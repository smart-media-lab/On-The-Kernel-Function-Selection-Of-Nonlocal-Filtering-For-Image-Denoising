function Filtering_ICMLC_2008_main
%
% MATLAB implementation of the paper J. Tian, W. Yu and S. Xie, 
% "On The Kernel Function Selection Of Nonlocal Filtering For Image Denoising,"
% Proc. Int. Conf. on Machine Learning and Cybernetics, pp. 2964-2969, Jul. 2008,
% Kunming, China.
%   
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

% Read the original image                
img_orig  = double(imread('lena128.bmp'));

% Adding noise
noise_mu = 0;
noise_sigma = 10;
img_noise = img_orig + randn(size(img_orig)) .* noise_sigma + noise_mu;

% Parameters of non-local filter        
M = 3;          % in equation (3)
lambda = 500;   % in equation (5)-(10)
L = 15;         % in equation (2)
nKernel = 1;    % 0:original; 1:cosine; 2:flat; 3:gaussian; 4:turkey; 5:wave

% Perform image filtering
img_filtered =  func_icmlc08_non_local_filtering(img_noise, L, M, lambda, nKernel);

% Compute PSNR performance
MSE = sum(sum((img_orig-img_filtered).^2));
MSE = MSE/size(img_orig,1)/size(img_orig,2);
nPSNR = 10*log10(255*255/MSE);
fprintf('PSNR=%.2fdB.\n', nPSNR);


%-------------------------------------------------------------------------
%------------------------------Inner Function ----------------------------
%-------------------------------------------------------------------------
function result = func_icmlc08_non_local_filtering(img_noise, L, M, h, nKernel)
%
% MATLAB implementation of the paper J. Tian, W. Yu and S. Xie, 
% "On The Kernel Function Selection Of Nonlocal Filtering For Image Denoising,"
% Proc. Int. Conf. on Machine Learning and Cybernetics, pp. 2964-2969, Jul. 2008,
% Kunming, China.
%
%  img_noise: image to be filtered
%  L: radio of search window
%  M: radio of similarity window
%  h: degree of filtering
%  nKernel: selection of kernel function
%  result: filtered image
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Size of the image
[nrow, ncol]=size(img_noise);

% Memory for the result
result=zeros(nrow,ncol);

% Replicate the boundaries of the input image
img_noise2 = padarray(img_noise,[M M],'symmetric');

% Used mask
% mask = make_kernel(M);

mask=zeros(2*M+1,2*M+1);

for d=1:M    
    value= 1/(2*d+1)^2 ;    
    for i=-d:d
        for j=-d:d
            mask(M+1-i,M+1-j)= mask(M+1-i,M+1-j) + value ;
        end
    end
end
mask = mask ./ M;
mask = mask / sum(sum(mask));

nSearch_step = 2; 

for i=1:nrow
    for j=1:ncol

        i1 = i+ M;
        j1 = j+ M;

        W1= img_noise2(i1-M:i1+M , j1-M:j1+M);
        wmax=0; 
        average=0;
        sweight=0;
        rmin = max(i1-L,M+1);
        rmax = min(i1+L,nrow+M);
        smin = max(j1-L,M+1);
        smax = min(j1+L,ncol+M);

        for r=rmin: nSearch_step:rmax
            for s=smin: nSearch_step:smax

                if(r==i1 && s==j1) continue; end;

                W2= img_noise2(r-M:r+M , s-M:s+M);                
                d = sum(sum(mask.*(W1-W2).*(W1-W2)));

                switch nKernel
                    case 0  % exponential function
                        w = exp(-d/h);      
                    case 1  % cos function
                        if abs(d)<=h           
                            w = cos(pi*d/2/h);
                        else
                            w = 0;
                        end
                    case 2  % flat function
                        if abs(d)<=h           
                            w = 1/d;
                        else
                            w = 0;
                        end                          
                    case 3  % Gaussian kernel
                        w = exp(-d*d/2/h/h);      
                    case 4  % tukey bi-wright function
                        if abs(d)<=h           
                            w = ((1-d*d/h/h)^2)/2;
                        else
                            w = 0;
                        end
                    case 5  % wave function
                        if abs(d)<=h           
                            w = sin(pi*d/h)/pi/d/h;
                        else
                            w = 0;
                        end
                end

                if w>wmax
                    wmax=w;
                end
                sweight = sweight + w;
                average = average + w*img_noise2(r,s);
            end
        end

        average = average + wmax*img_noise2(i1,j1);
        sweight = sweight + wmax;

        if sweight > 0
            result(i,j) = average / sweight;
        else
            result(i,j) = img_noise(i,j);
        end             
    end
end




