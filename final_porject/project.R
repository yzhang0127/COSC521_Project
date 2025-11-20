library(igraph)

cm_df <- read.csv("confusion_matrix_extracted.csv",row.names = 1,check.names = FALSE)

# replace NAs with 0
cm_df[is.na(cm_df)] <- 0
cm <- as.matrix(data.matrix(cm_df))

# remove 'Background'
bg_idx <- which(tolower(rownames(cm)) == "background")
if (length(bg_idx) > 0) {
  cm <- cm[-bg_idx, -bg_idx, drop = FALSE]
}

labels <- colnames(cm)
#sanity check
cat("Matrix size:", nrow(cm), "x", ncol(cm), "\n")
cat("First labels:", paste(head(labels, 5), collapse = ", "), "\n")

# remove correct predictions
diag(cm) <- 0

# build weighted directed edge list from non-zero cells
idx <- which(cm > 0, arr.ind = TRUE)
edges <- data.frame(
  source = rownames(cm)[idx[, "row"]],
  target = colnames(cm)[idx[, "col"]],
  weight = cm[idx]
)
g <- graph_from_data_frame(edges, directed = TRUE, vertices = data.frame(name = labels))
plot(g,
     vertex.label.cex = 0.8,
     vertex.label.color = "black",
     vertex.frame.color = "white",
     edge.curved = 0.1,
     main = "Error Propagation Network (YOLOv8 Confusion Matrix)"
)
print(gsize(g))

#Centrality metrics
#how many total errors each node participates in
str_all <- strength(g, mode = "all", weights = E(g)$weight)
str_in   <- strength(g, mode = "in",   weights = E(g)$weight)
str_out  <- strength(g, mode = "out",  weights = E(g)$weight)
# Betweenness
#Larger weights = Stronger ties
w <- E(g)$weight
E(g)$inv_w <- 1 / w
btw <- betweenness(g, directed = TRUE, weights = E(g)$inv_w, normalized = TRUE)

#PageRank
pr <- page_rank(g, directed = TRUE, weights = E(g)$weight, damping = 0.85)$vector

# Clustering/communities (use undirected collapse with summed weights)
g_u <- as_undirected(g, mode = "collapse", edge.attr.comb = list(weight = "sum"))
comm <- cluster_louvain(g_u, weights = E(g_u)$weight)
modularity_val <- modularity(comm)

#Show top nodes by metrics
show_top <- function(vec, k = 5, title = "Top") {
  ord <- order(vec, decreasing = TRUE)
  cat("\n", title, "\n", sep = "")
  print(round(vec[ord][1:min(k, length(vec))], 3))
}

show_top(str_all, 5, "Top by weighted degree (strength, in+out)")
show_top(str_in,  5, "Top by weighted in-strength (incoming errors)")
show_top(str_out, 5, "Top by weighted out-strength (outgoing misclassifications)")
show_top(btw,     5, "Top by betweenness (bridges)")
show_top(pr,     5, "Top by PageRank centrality (influence)")

cat("\nCommunity modularity (undirected collapse):", round(modularity_val, 3), "\n")

###############Visualization###############
top_k <- 10
save_bar <- function(vec, title, fname) {
  vec <- sort(vec, decreasing = TRUE)
  vec <- vec[seq_len(min(top_k, length(vec)))]
  png(fname, width = 1400, height = 1000, res = 200)
  par(mar = c(8,5,4,2))
  barplot(
    rev(vec), horiz = TRUE, las = 1,
    col = "grey70", border = NA,
    main = title, xlab = "", cex.names = 0.9
  )
  dev.off()
}
save_bar(str_all, "Top by Weighted Degree (Strength, in+out)", "fig3a_strength_top10.png")
save_bar(btw,     "Top by Betweenness (Bridging Roles)",        "fig3b_betweenness_top10.png")
save_bar(pr,      "Top by PageRank (Influence)",                "fig3c_pagerank_top10.png")
save_bar(str_in,  "Top by Weighted In-Strength (Incoming Errors)", 
         "fig3d_strength_in_top10.png")
save_bar(str_out, "Top by Weighted Out-Strength (Outgoing Errors)", 
         "fig3e_strength_out_top10.png")
#Louvain graph visual

g_u <- as_undirected(g, mode = "collapse", edge.attr.comb = list(weight = "sum"))

comm <- cluster_louvain(g_u, weights = E(g_u)$weight)
V(g)$community <- membership(comm)[match(V(g)$name, names(membership(comm)))]

set.seed(7)
L2 <- layout_with_fr(g_u, weights = E(g_u)$weight)
cols <- grDevices::rainbow(max(V(g)$community, na.rm = TRUE))

png("fig4_communities.png", width = 1800, height = 1400, res = 200)
par(mar = c(1,1,4,1))
plot(
  g_u, layout = L2,
  vertex.color = cols[V(g)$community],
  vertex.size = 12,
  vertex.label.cex = 0.8,
  vertex.label.color = "black",
  edge.color = rgb(0,0,0,0.35),
  edge.width = 1 + 3*(E(g_u)$weight/max(E(g_u)$weight)),
  main = paste0("Communities (Louvain) — Modularity = ", sprintf("%.3f", modularity(comm)))
)
legend("topleft", legend = paste("C", sort(unique(V(g)$community))), pch = 19,
       col = cols[sort(unique(V(g)$community))], pt.cex = 1.2, bty = "n")
dev.off()

#degree distribution
deg_in  <- degree(g, mode = "in")
deg_out <- degree(g, mode = "out")

png("fig5_degree_distribution.png", width = 1400, height = 1000, res = 200)
par(mfrow = c(1,2), mar = c(5,5,4,2))
hist(deg_in,  breaks = seq(-0.5, max(deg_in)+0.5, by = 1),
     col = "grey70", border = "white", main = "In-degree", xlab = "Degree")
hist(deg_out, breaks = seq(-0.5, max(deg_out)+0.5, by = 1),
     col = "grey70", border = "white", main = "Out-degree", xlab = "Degree")
dev.off()
