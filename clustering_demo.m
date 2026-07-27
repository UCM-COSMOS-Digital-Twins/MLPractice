%% CLUSTERING_DEMO
% Compares 3 unsupervised clustering algorithms on the Fisher Iris
% dataset (built-in, no download needed) -- with the species LABELS
% HIDDEN from the algorithms. We only reveal them at the end to check
% how well each method rediscovered the natural groups.
%   1) K-Means
%   2) DBSCAN
%   3) Hierarchical (Agglomerative) Clustering
%
% We also inject one synthetic "outlier" point to show how each method
% handles it differently -- this is the most important teaching moment
% in this script, so read the notes below carefully.
%
% Requires: Statistics and Machine Learning Toolbox

clear; clc; close all;

%% Load data (drop labels -- pretend we don't know the species!)
load fisheriris
X = meas(:, 3:4);              % Petal Length, Petal Width (most visual pair)
trueLabels = species;          % kept aside only for final comparison

% Standardize (important for distance-based methods)
Xs = (X - mean(X)) ./ std(X);

% Inject one clear outlier far from all clusters
Xs = [Xs; 6 6];

%% 1) K-Means (must choose K in advance)
rng(1);
K = 3;
[idxKM, C] = kmeans(Xs, K, 'Replicates', 5);

figure('Name','Clustering Comparison','Position',[100 100 1500 420]);
subplot(1,4,1);
gscatter(Xs(:,1), Xs(:,2), idxKM);
hold on; plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 12, 'LineWidth', 3);
title({'K-Means (K=3)','forced to put the outlier in a real cluster'});
xlabel('Petal Length (std.)'); ylabel('Petal Width (std.)');

%% 2) DBSCAN -- choose epsilon with the standard "k-distance knee" method
% Rather than guessing a radius, the standard data-driven approach is:
% for each point, compute its distance to its k-th nearest neighbor
% (k = minPts), sort those distances, and look for the "knee" where the
% curve bends sharply upward -- points past the knee are in sparse
% regions (likely noise/outliers). We detect the knee programmatically
% as the point of maximum distance from the line connecting the first
% and last points of the sorted curve.
minPts = 5;
D = pdist2(Xs, Xs);
Dsorted = sort(D, 2);
kDist = sort(Dsorted(:, minPts + 1));      % +1 skips distance to self (0)

xs_ = (1:numel(kDist))';
lineVec = [xs_(end) - xs_(1), kDist(end) - kDist(1)];
lineVec = lineVec / norm(lineVec);
vecFromFirst = [xs_ - xs_(1), kDist - kDist(1)];
proj = vecFromFirst * lineVec';
distToLine = sqrt(sum((vecFromFirst - proj * lineVec).^2, 2));
[~, kneeIdx] = max(distToLine);
epsilon = kDist(kneeIdx);

figure('Name','DBSCAN epsilon selection','Position',[100 550 500 400]);
plot(kDist, 'LineWidth', 1.5); hold on;
plot(kneeIdx, epsilon, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('Points sorted by distance'); ylabel(sprintf('Distance to %d-th nearest neighbor', minPts));
title(sprintf('k-Distance Plot: knee chosen at eps = %.3f', epsilon));
legend('sorted k-distances', 'chosen eps (knee)', 'Location', 'northwest');
grid on;

idxDB = dbscan(Xs, epsilon, minPts);   % label -1 = noise / outlier

figure(1);  % return to the comparison figure
subplot(1,4,2);
gscatter(Xs(:,1), Xs(:,2), idxDB);
title({sprintf('DBSCAN (eps=%.2f, minPts=%d)', epsilon, minPts), ...
       'outlier correctly flagged as noise (-1)'});
xlabel('Petal Length (std.)'); ylabel('Petal Width (std.)');

%% 3) Hierarchical (Agglomerative) Clustering
% KEY TEACHING POINT: hierarchical clustering has NO concept of "noise" --
% every point must belong to some cluster. If we cut the tree at 3
% clusters (matching the 3 species), the single extreme outlier is so far
% from everything else that Ward's method isolates it FIRST, using up one
% of our 3 clusters on a single point -- and merging two real species
% together in the process. DBSCAN doesn't have this problem because it
% can label sparse points as noise instead of forcing them into a
% cluster. To recover species-like groups with hierarchical clustering
% when an outlier is present, you typically need K+1 clusters (one extra
% "slot" for the outlier). We show BOTH cuts side by side.
linkageTree = linkage(Xs, 'ward', 'euclidean');

idxHC3 = cluster(linkageTree, 'maxclust', 3);
subplot(1,4,3);
gscatter(Xs(:,1), Xs(:,2), idxHC3);
title({'Hierarchical, maxclust=3', 'outlier consumes a whole cluster'});
xlabel('Petal Length (std.)'); ylabel('Petal Width (std.)');

idxHC4 = cluster(linkageTree, 'maxclust', 4);
subplot(1,4,4);
gscatter(Xs(:,1), Xs(:,2), idxHC4);
title({'Hierarchical, maxclust=4', 'one extra cluster frees up species grouping'});
xlabel('Petal Length (std.)'); ylabel('Petal Width (std.)');

sgtitle('Four Clustering Results, Same Data (outlier planted at [6,6])');

%% ---- The dendrogram: hierarchical clustering's signature visual ----
figure('Name','Dendrogram','Position',[650 550 900 400]);
dendrogram(linkageTree, 0);   % 0 = show all leaves, no truncation
title('Hierarchical Clustering Dendrogram: The Full "Family Tree" of the Data');
xlabel('Sample index'); ylabel('Linkage distance (Ward)');
% Discussion: ask students where THEY would "cut" this tree, and why --
% point out that the very first, tallest split is the outlier splitting
% off from everything else.

%% ---- Reveal the true species labels for comparison ----
figure('Name','Ground Truth','Position',[1050 100 500 400]);
gscatter(Xs(1:150,1), Xs(1:150,2), trueLabels);
title('Ground Truth Species (hidden from all algorithms above)');
xlabel('Petal Length (std.)'); ylabel('Petal Width (std.)');

%% ---- Discussion prompts for class ----
% 1) K-means was FORCED to assign the planted outlier at [6,6] to one of
%    3 clusters, which can drag that cluster's centroid off target.
%    DBSCAN was allowed to call it "noise" (-1) instead. Which behavior
%    would you want for fraud detection? For customer segments?
% 2) Compare Hierarchical at maxclust=3 vs. maxclust=4. Why does adding
%    just ONE more allowed cluster suddenly let it recover species-like
%    groups? What does that tell you about hierarchical clustering's
%    blind spot for outliers compared to DBSCAN?
% 3) In the ground-truth plot, notice that two species (versicolor and
%    virginica) genuinely overlap in petal measurements -- this isn't a
%    bug in any algorithm; it's a real property of this dataset. Which
%    of today's 3 methods would you trust most to tell you the "correct"
%    number of clusters when you don't already know the answer, and why?
