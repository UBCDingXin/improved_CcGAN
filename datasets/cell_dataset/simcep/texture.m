%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TEXTURE Function for generating 2D Perlin noise
% Input:  (1) size of the output image
%         (2) persistence
%         (3) first octave
%         (4) last ocrave
%         (5) texture bias
% Output: (1) simulated texture
%
% (C) 22.2.2007 Antti Lehmussola
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[out] = texture(sz,p,n1,n2,b)

out = zeros(sz);

if n1 < 1
    n1 = 1;
end

for ind = n1:(n2+n1)-1
    
    f = power(2,ind);
    
    amp = power(p,ind);

    randkuva = rand(f);
    randkuva = [randkuva(:,1) randkuva randkuva(:,end)];
    randkuva = [randkuva(1,:); randkuva; randkuva(end,:)];
    
    H = ones(3)/9;
    randkuva = conv2(randkuva,H,'valid');
    
    y = linspace(1,size(randkuva,1),sz(1));
    x = linspace(1,size(randkuva,2),sz(2))';
    
    if f < 3
        method = 'bilinear';
    else
        method = 'bicubic';
    end
    
    interpkuva = interp2(randkuva,x,y,method);

    out = out + interpkuva*amp;
end

out = out - min(out(:));
out = out./max(out(:));
out = b + (1-b)*out;
