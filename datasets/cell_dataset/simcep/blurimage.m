%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function for calling blurring mex-function
% IN:   I       image
%       lev     level (or "variance coefficient") for each pixel,
%               assumed to be between [0,1]
%       win     window length (default = 9)
%       mi      minimum value for variance (default = min(lev(:)))
%       ma      maximum value for variance (default = max(lev(:)))
% OUT:  C       blurred image
%
% (C) Pekka Ruusuvuori, 1.2.2006
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function C = blurimage(I,lev,win,mi,ma);

%Padding
l = floor(win/2)+1;
I = [repmat(I(:,1),1,l) I  repmat(I(:,end),1,l)];
I = [repmat(I(1,:),l,1); I; repmat(I(end,:),l,1)];

lev = [repmat(lev(:,1),1,l) lev  repmat(lev(:,end),1,l)];
lev = [repmat(lev(1,:),l,1); lev; repmat(lev(end,:),l,1)];

C = blur(I,lev,win,mi,ma);

C = C(l+1:end-l,l+1:end-l);

