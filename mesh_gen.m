% Generate a quasi-random triangulation of domain [ax,bx]X[ay,by]
% with N points

N = 54;
ax = -3;
bx = 3;
ay = -3;
by = 3;
P = [];

% Generate domain boundary points
bndN = round(sqrt(N));
bndHx = (bx-ax)/(bndN-1);
bndHy = (by-ay)/(bndN-1);
for i = 0:(bndN-1)
    P = [P;ax,ay+i*bndHy];
    P = [P;bx,ay+i*bndHy];
    P = [P;ax+i*bndHx,ay];
    P = [P;ax+i*bndHx,by];
end
P = unique(P,'rows');

% Generate interior domain points
N = N - size(P,1);

% Random triangulation from structured interior grid
Nd = FindClosestFactorization(N);
Nx = Nd(1)+1;
Ny = Nd(2)+1;
Hx = (bx-ax)/Nx;
Hy = (by-ay)/Ny;
for i = 1:(Nx-1)
    for j = 1:(Ny-1)
        dx = 0.3*(2*rand(1)-1)*Hx;
        xi = ax+i*Hx+dx;
        dy = 0.3*(2*rand(1)-1)*Hy;
        yi = ay+j*Hy+dy;
        P = [P;xi,yi];
    end
end

% Random triangulation from Halton or Sobol set
% rng default  % For reproducibility
% p = haltonset(2,'Skip',1e3,'Leap',1e2);
% p = scramble(p,'RR2');
% % p = sobolset(2,'Skip',1e3,'Leap',1e2);
% % p = scramble(p,'MatousekAffineOwen');
% X0 = net(p,N);
% for i = 1:size(X0,1)
%     X0(i,1) = (ax+bndHx/2) + (bx-ax-bndHx)*X0(i,1);
%     X0(i,2) = (ay+bndHy/2) + (by-ay-bndHy)*X0(i,2);
% end
% P = [P;X0];

% Generate Gamma boundary points
% G: two circles
Gx = [];
GN = ceil(bndN/2);
for i = 1:GN
    xp = 1+(1/2)*cos(pi*(2*i-1)/(2*GN));
    Gx = [Gx,xp];
end
Gy1 = sqrt(0.25-(Gx-1).^2);
Gy2 = -Gy1;
P = [P;Gx',Gy1'];
P = [P;Gx',Gy2'];
Gx = [];
for i = 1:GN
    xp = (-sqrt(1.5))+(1/2)*cos(pi*(2*i-1)/(2*GN));
    Gx = [Gx,xp];
end
Gy1 = sqrt(0.25-(Gx+sqrt(1.5)).^2);
Gy2 = -Gy1;
P = [P;Gx',Gy1'];
P = [P;Gx',Gy2'];
P = unique(P,'rows');

subplot(4,4,13)
scatter(P(:,1),P(:,2),'filled')
hold on
DT = delaunay(P);
triplot(DT,P(:,1),P(:,2))
c = size(P,1);
c = num2str(c) + " nodes";
title({'{\bf Quasi-Random Delaunay Triangulation}, ',c})
xlim([ax bx])
ylim([ay by])
axis square
hold off

% Refinement
[Pr,DTr] = refineXY(P,DT);
subplot(4,4,10)
scatter(Pr(:,1),Pr(:,2),'filled')
hold on
triplot(DTr,Pr(:,1),Pr(:,2))
c = size(Pr,1);
title(['{\bf XY Refined}, ',num2str(c),' nodes'])
xlim([ax bx])
ylim([ay by])
axis square
hold off

[PrXXY,DTrXXY] = refineX(Pr,DTr);
subplot(4,4,11)
scatter(PrXXY(:,1),PrXXY(:,2),'filled')
hold on
triplot(DTrXXY,PrXXY(:,1),PrXXY(:,2))
c = size(PrXXY,1);
title(['{\bf XXY Refined}, ',num2str(c),' nodes'])
xlim([ax bx])
ylim([ay by])
axis square
hold off

[Prr,DTrr] = refineXY(Pr,DTr);
subplot(4,4,7)
scatter(Prr(:,1),Prr(:,2),'filled')
hold on
triplot(DTrr,Prr(:,1),Prr(:,2))
c = size(Prr,1);
title(['{\bf XXYY Refined}, ',num2str(c),' nodes'])
xlim([ax bx])
ylim([ay by])
axis square
hold off

[Prrr,DTrrr] = refineXY(Prr,DTrr);
subplot(4,4,4)
scatter(Prrr(:,1),Prrr(:,2),'filled')
hold on
triplot(DTrrr,Prrr(:,1),Prrr(:,2))
c = size(Prrr,1);
title(['{\bf XXXYYY Refined}, ',num2str(c),' nodes'])
xlim([ax bx])
ylim([ay by])
axis square
hold off

% Refinement in x
[PrX,DTrX] = refineX(P,DT);
subplot(4,4,14)
scatter(PrX(:,1),PrX(:,2),'filled')
hold on
triplot(DTrX,PrX(:,1),PrX(:,2))
c = size(PrX,1);
title(['{\bf X Refined}, ',num2str(c),' nodes'])
xlim([ax bx])
ylim([ay by])
axis square
hold off

[PrXX,DTrXX] = refineX(PrX,DTrX);
subplot(4,4,15)
scatter(PrXX(:,1),PrXX(:,2),'filled')
hold on
triplot(DTrXX,PrXX(:,1),PrXX(:,2))
c = size(PrXX,1);
title(['{\bf XX Refined}, ',num2str(c),' nodes'])
xlim([ax bx])
ylim([ay by])
axis square
hold off

[PrXXX,DTrXXX] = refineX(PrXX,DTrXX);
subplot(4,4,16)
scatter(PrXXX(:,1),PrXXX(:,2),'filled')
hold on
triplot(DTrXXX,PrXXX(:,1),PrXXX(:,2))
c = size(PrXXX,1);
title(['{\bf XXX Refined}, ',num2str(c),' nodes'])
xlim([ax bx])
ylim([ay by])
axis square
hold off

% Refinement in y
[PrY,DTrY] = refineY(P,DT);
subplot(4,4,9)
scatter(PrY(:,1),PrY(:,2),'filled')
hold on
triplot(DTrY,PrY(:,1),PrY(:,2))
c = size(PrY,1);
title(['{\bf Y Refined}, ',num2str(c),' nodes'])
xlim([ax bx])
ylim([ay by])
axis square
hold off

[PrYY,DTrYY] = refineY(PrY,DTrY);
subplot(4,4,5)
scatter(PrYY(:,1),PrYY(:,2),'filled')
hold on
triplot(DTrYY,PrYY(:,1),PrYY(:,2))
c = size(PrYY,1);
title(['{\bf YY Refined}, ',num2str(c),' nodes'])
xlim([ax bx])
ylim([ay by])
axis square
hold off

[PrYYY,DTrYYY] = refineY(PrYY,DTrYY);
subplot(4,4,1)
scatter(PrYYY(:,1),PrYYY(:,2),'filled')
hold on
triplot(DTrYYY,PrYYY(:,1),PrYYY(:,2))
c = size(PrYYY,1);
title(['{\bf YYY Refined}, ',num2str(c),' nodes'])
xlim([ax bx])
ylim([ay by])
axis square
hold off

[PrXYY,DTrXYY] = refineX(PrYY,DTrYY);
subplot(4,4,6)
scatter(PrXYY(:,1),PrXYY(:,2),'filled')
hold on
triplot(DTrXYY,PrXYY(:,1),PrXYY(:,2))
c = size(PrXYY,1);
title(['{\bf XYY Refined}, ',num2str(c),' nodes'])
xlim([ax bx])
ylim([ay by])
axis square
hold off



function factors = FindClosestFactorization(y)
    %FINDCLOSEFACTORS Given y, finds x1 and x2 such that x1*x2 = y and |x1-x2| is minimized.
    %   If y is not an integer, it will be rounded towards the nearest one.
    %
    %   Examples:
    %       Input: 12   Output: [3 4]
    %       Input: 39   Output: [3 13]
    %       Input: 4    Output: [2 2]
    %       Input: 7    Output: [1 7]
    %
    %   Author: Luke Gane
    %	Version: 1.0
    %   Last updated: 2016-05-21
    y = round(y(:));
    firstFactor = arrayfun(@FindFirstFactor, y);
    secondFactor = y./firstFactor;
    factors = [firstFactor secondFactor];
end

function firstFactor = FindFirstFactor(value)
    firstFactor = floor(sqrt(value));
    while (mod(value, firstFactor) ~= 0)
        firstFactor = firstFactor - 1;
    end
end

function [PR,DTR] = refineXY(Pt,DTt)
    PR = Pt;
    for i = 1:size(DTt,1)

        xmin = min(Pt(DTt(i,1),1),Pt(DTt(i,2),1));
        xmax = max(Pt(DTt(i,1),1),Pt(DTt(i,2),1));
        xn1 = xmin + (xmax-xmin)/3;
        xn2 = xmin + 2*(xmax-xmin)/3;
        ymin = min(Pt(DTt(i,1),2),Pt(DTt(i,2),2));
        ymax = max(Pt(DTt(i,1),2),Pt(DTt(i,2),2));
        yn1 = ymin + (ymax-ymin)/3;
        yn2 = ymin + 2*(ymax-ymin)/3;
        PR = [PR;xn1,yn1;xn2,yn2];

        xmin = min(Pt(DTt(i,2),1),Pt(DTt(i,3),1));
        xmax = max(Pt(DTt(i,2),1),Pt(DTt(i,3),1));
        xn1 = xmin + (xmax-xmin)/3;
        xn2 = xmin + 2*(xmax-xmin)/3;
        ymin = min(Pt(DTt(i,2),2),Pt(DTt(i,3),2));
        ymax = max(Pt(DTt(i,2),2),Pt(DTt(i,3),2));
        yn1 = ymin + (ymax-ymin)/3;
        yn2 = ymin + 2*(ymax-ymin)/3;
        PR = [PR;xn1,yn1;xn2,yn2];

        xmin = min(Pt(DTt(i,1),1),Pt(DTt(i,3),1));
        xmax = max(Pt(DTt(i,1),1),Pt(DTt(i,3),1));
        xn1 = xmin + (xmax-xmin)/3;
        xn2 = xmin + 2*(xmax-xmin)/3;
        ymin = min(Pt(DTt(i,1),2),Pt(DTt(i,3),2));
        ymax = max(Pt(DTt(i,1),2),Pt(DTt(i,3),2));
        yn1 = ymin + (ymax-ymin)/3;
        yn2 = ymin + 2*(ymax-ymin)/3;
        PR = [PR;xn1,yn1;xn2,yn2];

    end
    PR = unique(PR,'rows');
    DTR = delaunay(PR);
end

function [PR,DTR] = refineX(Pt,DTt)
        PR = Pt;
    for i = 1:size(DTt,1)

        xmin = min(Pt(DTt(i,1),1),Pt(DTt(i,2),1));
        xmax = max(Pt(DTt(i,1),1),Pt(DTt(i,2),1));
        xn1 = xmin + (xmax-xmin)/3;
        xn2 = xmin + 2*(xmax-xmin)/3;
        ymin = min(Pt(DTt(i,1),2),Pt(DTt(i,2),2));
        ymax = max(Pt(DTt(i,1),2),Pt(DTt(i,2),2));
        yn1 = ymin + (ymax-ymin)/3;
        yn2 = ymin + 2*(ymax-ymin)/3;
        if abs((Pt(DTt(i,1),1) - Pt(DTt(i,2),1))/(Pt(DTt(i,1),2) - Pt(DTt(i,2),2))) >= 1
            PR = [PR;xn1,yn1;xn2,yn2];
        end

        xmin = min(Pt(DTt(i,2),1),Pt(DTt(i,3),1));
        xmax = max(Pt(DTt(i,2),1),Pt(DTt(i,3),1));
        xn1 = xmin + (xmax-xmin)/3;
        xn2 = xmin + 2*(xmax-xmin)/3;
        ymin = min(Pt(DTt(i,2),2),Pt(DTt(i,3),2));
        ymax = max(Pt(DTt(i,2),2),Pt(DTt(i,3),2));
        yn1 = ymin + (ymax-ymin)/3;
        yn2 = ymin + 2*(ymax-ymin)/3;
        if abs((Pt(DTt(i,2),1) - Pt(DTt(i,3),1))/(Pt(DTt(i,2),2) - Pt(DTt(i,3),2))) >= 1
            PR = [PR;xn1,yn1;xn2,yn2];
        end

        xmin = min(Pt(DTt(i,1),1),Pt(DTt(i,3),1));
        xmax = max(Pt(DTt(i,1),1),Pt(DTt(i,3),1));
        xn1 = xmin + (xmax-xmin)/3;
        xn2 = xmin + 2*(xmax-xmin)/3;
        ymin = min(Pt(DTt(i,1),2),Pt(DTt(i,3),2));
        ymax = max(Pt(DTt(i,1),2),Pt(DTt(i,3),2));
        yn1 = ymin + (ymax-ymin)/3;
        yn2 = ymin + 2*(ymax-ymin)/3;
        if abs((Pt(DTt(i,1),1) - Pt(DTt(i,3),1))/(Pt(DTt(i,1),2) - Pt(DTt(i,3),2))) >= 1
            PR = [PR;xn1,yn1;xn2,yn2];
        end

    end
    PR = unique(PR,'rows');
    DTR = delaunay(PR);
end

function [PR,DTR] = refineY(Pt,DTt)
        PR = Pt;
    for i = 1:size(DTt,1)

        xmin = min(Pt(DTt(i,1),1),Pt(DTt(i,2),1));
        xmax = max(Pt(DTt(i,1),1),Pt(DTt(i,2),1));
        xn1 = xmin + (xmax-xmin)/3;
        xn2 = xmin + 2*(xmax-xmin)/3;
        ymin = min(Pt(DTt(i,1),2),Pt(DTt(i,2),2));
        ymax = max(Pt(DTt(i,1),2),Pt(DTt(i,2),2));
        yn1 = ymin + (ymax-ymin)/3;
        yn2 = ymin + 2*(ymax-ymin)/3;
        if abs((Pt(DTt(i,1),1) - Pt(DTt(i,2),1))/(Pt(DTt(i,1),2) - Pt(DTt(i,2),2))) < 1
            PR = [PR;xn1,yn1;xn2,yn2];
        end

        xmin = min(Pt(DTt(i,2),1),Pt(DTt(i,3),1));
        xmax = max(Pt(DTt(i,2),1),Pt(DTt(i,3),1));
        xn1 = xmin + (xmax-xmin)/3;
        xn2 = xmin + 2*(xmax-xmin)/3;
        ymin = min(Pt(DTt(i,2),2),Pt(DTt(i,3),2));
        ymax = max(Pt(DTt(i,2),2),Pt(DTt(i,3),2));
        yn1 = ymin + (ymax-ymin)/3;
        yn2 = ymin + 2*(ymax-ymin)/3;
        if abs((Pt(DTt(i,2),1) - Pt(DTt(i,3),1))/(Pt(DTt(i,2),2) - Pt(DTt(i,3),2))) < 1
            PR = [PR;xn1,yn1;xn2,yn2];
        end

        xmin = min(Pt(DTt(i,1),1),Pt(DTt(i,3),1));
        xmax = max(Pt(DTt(i,1),1),Pt(DTt(i,3),1));
        xn1 = xmin + (xmax-xmin)/3;
        xn2 = xmin + 2*(xmax-xmin)/3;
        ymin = min(Pt(DTt(i,1),2),Pt(DTt(i,3),2));
        ymax = max(Pt(DTt(i,1),2),Pt(DTt(i,3),2));
        yn1 = ymin + (ymax-ymin)/3;
        yn2 = ymin + 2*(ymax-ymin)/3;
        if abs((Pt(DTt(i,1),1) - Pt(DTt(i,3),1))/(Pt(DTt(i,1),2) - Pt(DTt(i,3),2))) < 1
            PR = [PR;xn1,yn1;xn2,yn2];
        end

    end
    PR = unique(PR,'rows');
    DTR = delaunay(PR);
end