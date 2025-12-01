#################################### DEEP LEARNING COMPUTER VISION MODELS FOR EARTH OBSERVATION APPLICATIONS ####################################

This document tracks the active development of my Machine Learning (ML) and Deep Learning (DL) library for geospatial applications spanning:

# Cadastre

# Earth Observation (EO)

    ✅ Change Detection (urban growth, deforestation, etc.)
    ✅ Classification (land cover, land use, etc.)
    ✅ Data (super-resolution, augmentation, etc.)
    ✅ Hyperspectral Analysis (crop health, mineralogy, etc.)
    ✅ Multi-scale Analysis (from trees to landscapes)
    ✅ Object Detection (cars, ships, trees, roofs, etc.)
    ✅ Radar Applications (deformation, subsidence, etc.)
    ✅ Segmentation (building footprints, water bodies, etc.)
    ✅ 3D Modeling (digital twins, DEMs, etc.)

# Geographic Information Systems (GIS)

# Global Navigation Satellite Systems (GNSS)

# Geosciences

# Land Management

# Real Estate Valuation

# Surveying


🌍 The scope extends beyond classical Computer Vision (CV) tasks to include:
- Error correction
- Graph-based learning for spatial networks
- Multi-modal and multi-sensor data fusion
- Physics-informed models for geospatial constraints
- Signal processing
- Time-series analysis and forecasting
- Other specialized geospatial domains


⚠️ As this library is an ongoing work, please note you may encounter:
- Documentation inconsistencies
- Duplicate or overlapping models
- Evolving APIs
- Incomplete modules
- Missing examples and tutorials
- Missing sections
- Partial implementations
- Performance variations between versions
- Temporary workarounds
- Testing gaps
- Transitional states
- Unstable dependencies


1. Convolutional Neural Networks (CNNs).

000. CNN - LeNet-5 (1998) - Pioneering foundational CNN; historically significant for early remote sensing research. Largely superseded for modern high-performance applications.
001. CNN - AlexNet (2012) - Pioneering deep CNN; historical importance but largely superseded for modern EO/RS applications due to shallow architecture
002. CNN - VGG (2014) - Simple uniform architecture; good feature extractor but parameter-heavy for large EO/RS imagery processing
003. CNN - Inception (2014, GoogLeNet) - Multi-scale feature extraction within layers; effective for EO/RS objects at different scales (vehicles to fields)
004. CNN - ResNet (2015, Residual Networks) - Revolutionary skip connections enable very deep networks; backbone for most modern EO/RS detection systems
005. CNN - U-Net (2015, Encoder-Decoder Paradigm) - Essential for semantic segmentation; dominant architecture for land cover mapping and building extraction
006. CNN - DenseNet (2017) - Dense connections enable feature reuse; parameter-efficient for limited EO/RS training data scenarios
007. CNN - FPN (2017, Feature Pyramid Network) - Multi-scale feature fusion; critical for detecting EO/RS objects from small vehicles to large agricultural fields
008. CNN - PSPNet (2017, Pyramid Scene Parsing Network) - Global context capture through pyramid pooling; excellent for large-area EO/RS scene understanding
009. CNN - Mask R-CNN (2017) - Instance segmentation capability; perfect for individual object extraction in dense urban EO/RS scenes
010. CNN - DeepLabV3+ (2018) - Atrous spatial pyramid pooling; state-of-the-art for multi-scale EO/RS semantic segmentation tasks
011. CNN - MobileNetV2 (2018) - Lightweight inverted residual architecture; ideal for edge deployment and real-time EO/RS applications
012. CNN - EfficientNet (2019) - Compound scaling optimization; provides best accuracy/efficiency trade-off for large-scale EO/RS processing
013. CNN - HRNet (2019, High-Resolution Network) - Maintains high-resolution features throughout; superior for precise boundary detection in EO/RS imagery
014. CNN - Rotated R-CNN (2021) - Specialized for oriented object detection; essential for ships, vehicles, and buildings in aerial imagery


2. Recurrent Neural Networks (RNNs).

015. RNN - Basic RNN (1980s, foundational) - Simple recurrent connections; limited modern EO/RS use due to vanishing gradient problems in long sequences
016. RNN - LSTM (1997, Long Short-Term Memory) - Gating mechanisms enable long-term memory; essential for multi-temporal EO/RS analysis and crop monitoring
017. RNN - GRU (2014, Gated Recurrent Unit) - Simplified LSTM with fewer parameters; efficient for satellite time series classification and phenology tracking
018. RNN - ConvLSTM (2015, Convolutional LSTM) - Critical for spatio-temporal data; combines CNN spatial processing with RNN temporal modeling for cloud tracking and change detection
019. RNN - Bidirectional LSTM (Bi-LSTM) (1997) - Processes sequences in both directions; superior for complete growing season analysis and climate trend modeling
020. RNN - Stacked LSTM/GRU (2010s) - Deep recurrent architectures; captures complex temporal patterns in long-term satellite image series
021. RNN - Attention-Augmented RNN (2018) - Combines attention mechanisms with recurrence; focuses on key time steps in multi-year environmental monitoring
022. RNN - Temporal Convolutional Network (TCN) (2018) - CNN-based temporal modeling; efficient alternative to RNNs for long satellite time series
023. RNN - PredRNN (2017) - Advanced spatio-temporal forecasting; excellent for weather prediction and disaster progression modeling
024. RNN - Eidetic 3D LSTM (2018) - Specialized for video and image sequences; effective for high-frequency satellite video analysis
025. RNN - Phased LSTM (2016) - Time-aware gating mechanism; perfect for irregularly sampled satellite data and multi-sensor fusion


3. Transformers (aka Attention-Based Models).
026. TF - Original (2017) - Foundational encoder-decoder architecture; basis for modern attention models but rarely used directly in EO/RS due to computational complexity
027. TF - Derivatives - GPT (Generative Pre-trained Transformer) - GPT-2, GPT-3, GPT-4 - Large-scale generative model; potential for EO/RS report generation and data interpretation
028. TF - Derivatives - GPT (Generative Pre-trained Transformer) - LLaMA - Efficient large language model; adaptable for EO/RS metadata processing
029. TF - Derivatives - GPT (Generative Pre-trained Transformer) - OPT - Open-source alternative; for custom EO/RS language-vision applications
030. TF - Derivatives - BERT (Bidirectional Encoder Representations from Transformers) - RoBERTa - Optimized BERT variant; robust EO/RS text-data alignment
031. TF - Derivatives - BERT (Bidirectional Encoder Representations from Transformers) - ELECTRA - Sample-efficient pre-training; valuable for limited EO/RS labeled data
032. TF - ViT (2020, Vision Transformer) - Pure TF for images; state-of-the-art for EO/RS scene classification and land cover mapping
033. TF - Swin Transformer (2021) - Hierarchical shifting windows; dominant architecture for high-resolution EO/RS imagery analysis
034. TF - Longformer (2020) - Linear attention scaling; ideal for long satellite time series and multi-temporal analysis
035. TF - Performer (2021) - Linear approximate attention; efficient processing of large EO/RS scenes
036. TF - Remote Sensing Transformer (EO/RSFormer) - Specifically designed for EO/RS image characteristics and geometric properties
037. TF - Spectral-Spatial Transformer - Optimized for hyperspectral image analysis and band relationships
038. TF - GeoTransformer - Incorporates geographic coordinates and spatial constraints
039. TF - Cross-modal Transformer - Fuses optical, SAR, and LiDAR data through cross-attention mechanisms
040. TF - Multi-temporal Vision Transformer - Specifically designed for bi-temporal and multi-temporal change detection
041. TF - Vision-Language Transformer - Connects EO/RS imagery with textual descriptions for automated labeling
042. TF - ChangeFormer - Transformer-based architecture specifically for remote sensing change detection
043. TF - BIT (Bi-Temporal Transformer) - Specialized for before-after image comparison and change analysis
044. TF - STANet (Spatial-Temporal Attention) - Combines spatial and temporal attention for change monitoring
045. TF - Twins Transformer - Spatially separable attention; efficient for large-area EO/RS mapping
046. TF - CrossFormer - Multi-scale attention; handles EO/RS objects at different scales
047. TF - MobileViT - Lightweight vision TF; suitable for edge deployment in EO/RS


4. State-Space Models (SSMs).
048. SSM - Structured State Space Sequence Model (S4) - Foundational modern SSM architecture; enables efficient parallel scan for long-sequence satellite time series analysis
049. SSM - Mamba (2023) - Leading architecture with data-dependent selection; emerging as Transformer alternative for long EO/RS temporal sequences and efficient high-resolution processing
050. SSM - Vision Mamba (2024) - Mamba adapted for visual tasks; promising for large-scale EO/RS image classification and segmentation
051. SSM - VMamba - Visual state space model; efficient alternative to Vision Transformers for high-resolution satellite imagery
052. SSM - Selective State Spaces (S6) - Mamba's core mechanism; enables content-aware processing of EO/RS temporal patterns
053. SSM - Mamba-2 - Improved formulation; better performance on multi-spectral and hyperspectral sequences
054. SSM - Spatial Mamba - Extends SSMs to 2D spatial data; potential for direct EO/RS image processing
055. SSM - Hybrid SSM-Transformers - Combines SSM efficiency with Transformer expressivity; ideal for complex EO/RS multi-modal tasks
056. SSM - Liquid S4 - Continuous-time SSM variant; suitable for irregularly sampled satellite data
057. SSM - S4D - Diagonal state space model; efficient for seasonal pattern analysis in EO/RS time series


5. Generative Models.
058. GM - Generative Adversarial Networks (2014, GAN) - Adversarial training framework; widely used for EO/RS data augmentation, super-resolution, and domain adaptation
059. GM - Denoising Diffusion Probabilistic Models (2020, DDPM) - Foundational diffusion architecture; basis for modern EO/RS generative applications
060. GM - Denoising Diffusion Implicit Models (2020, DDIM) - Accelerated sampling; enables faster inference for EO/RS image generation tasks
061. GM - pix2pix (2017) - Paired image-to-image translation; essential for EO/RS tasks like SAR-to-optical translation and cloud removal
062. GM - CycleGAN (2017) - Unpaired domain adaptation; critical for cross-sensor EO/RS data harmonization and seasonal translation
063. GM - StyleGAN2/3 (2020) - High-quality synthesis; generates realistic EO/RS imagery for data augmentation
064. GM - SinGAN (2020) - Single-image training; useful for limited EO/RS data scenarios and texture synthesis
065. GM - Stable Diffusion (2022) - Latent space diffusion; dominant architecture for practical EO/RS image generation and editing
066. GM - Latent Diffusion Models (LDM) - Efficient diffusion in compressed space; ideal for high-resolution EO/RS imagery
067. GM - ControlNet (2023) - Conditional control; enables precise EO/RS image generation with spatial constraints
068. GM - InstructPix2Pix - Instruction-based editing; potential for interactive EO/RS image modification
069. GM - ChangeGAN - Specialized for change detection data augmentation; generates realistic before-after image pairs
070. GM - CloudGAN - Dedicated cloud removal and synthesis; improves cloud-affected EO/RS data utility
071. GM - SeasonGAN - Cross-seasonal image translation; generates off-season EO/RS imagery
072. GM - Super-Resolution GANs (SRGAN, ESRGAN) - Enhanced image resolution; critical for improving low-res satellite data
073. GM - VAE (2013) - Probabilistic latent space; useful for EO/RS data compression and anomaly detection
074. GM - VQ-VAE (2017) - Vector quantized latent space; enables discrete representation learning for EO/RS imagery
075. GM - NVAE (2020) - Hierarchical VAE; captures multi-scale EO/RS image structure
076. GM - RealNVP (2016) - Invertible transformations; exact density estimation for EO/RS data
077. GM - Glow (2018) - Generative flow models; high-quality EO/RS image synthesis with tractable likelihood


6. Graph Neural Networks (GNNs).
078. GNN - Graph Convolutional Network (GCN) - Spectral graph convolutions; foundational for node classification in EO/RS spatial networks
079. GNN - Graph Attention Network (GAT) - Attention-based neighborhood aggregation; dynamic importance weighting for EO/RS spatial relationships
080. GNN - Graph Neural Networks (GNNs) - General framework; umbrella term for all graph-based deep learning architectures
081. GNN - GraphSAGE (2017) - Inductive learning; scalable for large EO/RS geographic networks and dynamic urban systems
082. GNN - Graph U-Net (2019) - Graph pooling and unpooling; hierarchical segmentation of irregular EO/RS regions
083. GNN - PointNet/PointNet++ (2017) - Critical: Direct 3D point cloud processing; essential for LiDAR data and building reconstruction
084. GNN - Dynamic Graph CNN (DGCNN) - Edge convolution; adapts to local point cloud geometry for EO/RS 3D analysis
085. GNN - Spatio-Temporal GNN (ST-GNN) - Models both spatial and temporal dependencies; ideal for urban growth monitoring
086. GNN - EvolveGCN (2020) - Adapts graph structure over time; perfect for dynamic EO/RS phenomena like deforestation
087. GNN - Temporal Graph Networks (TGN) - Continuous-time dynamics; suitable for irregular EO/RS observation sequences
088. GNN - Graph Transformer (2019) - Self-attention on graphs; captures long-range dependencies in large EO/RS regions
089. GNN - Multi-scale GNNs - Hierarchical graph representations; handles different spatial scales in EO/RS analysis
090. GNN - Hypergraph Networks - Beyond pairwise relationships; models complex multi-way EO/RS interactions
091. GNN - Geographic GNNs - Incorporates spatial autocorrelation and geographic constraints
092. GNN - Spectral-Spatial GNNs - Combines graph learning with spectral information for hyperspectral data
093. GNN - Cross-modal Graph Networks - Fuses multiple EO/RS data sources (optical, SAR, LiDAR) via graph structures
094. GNN - Road Network GNNs - Specialized for transportation network analysis and traffic flow prediction
095. GNN - Message Passing Neural Networks (MPNN) - General framework; flexible for custom EO/RS graph structures
096. GNN - Graph Isomorphism Networks (GIN) - More expressive than GCN; better for complex EO/RS graph topology
097. GNN - Diffusion-Convolutional Neural Networks (DCNN) - Diffusion-based propagation; captures multi-hop EO/RS dependencies


7. Hybrid Architectures.
098. Hybrid - Jamba (Transformer-SSM Hybrid) - Combines Transformer blocks with Mamba (SSM) blocks; emerging for efficient long-sequence EO/RS time series with global attention capabilities
099. Hybrid - Swin U-Net - Combines Swin Transformer with U-Net structure; dominant for high-resolution EO/RS segmentation with global context
100. Hybrid - ConvLSTM/ConvGRU - Combines CNN spatial feature extraction with RNN temporal modeling; foundational for EO/RS spatio-temporal forecasting
101. Hybrid - CMT (Convolutional Vision Transformer) - Hybrid convolution-attention; efficient local-global feature fusion for EO/RS imagery
102. Hybrid - CoAtNet (Convolution + Attention) - Staged integration of CNNs and Transformers; optimal accuracy/efficiency for EO/RS classification
103. Hybrid - MobileViT - Lightweight mobile vision transformer; efficient deployment for edge EO/RS applications
104. Hybrid - CrossFormer - Multi-scale attention with convolution; handles varied object sizes in EO/RS scenes
105. Hybrid - Multi-modal Transformer-CNN - Fuses optical, SAR, and LiDAR through hybrid encoders; comprehensive EO/RS scene understanding
106. Hybrid - Cross-attention Fusion Networks - Uses cross-attention between different EO/RS data modalities; effective for sensor fusion
107. Hybrid - Tensor Fusion Networks - Multi-dimensional feature fusion; integrates spectral, spatial, and temporal EO/RS data
108. Hybrid - Transformer-LSTM Hybrids - Combines long-range attention with temporal modeling; ideal for climate time series analysis
109. Hybrid - 3D CNN-Transformer - Volumetric processing with attention; suitable for video satellite data and 3D EO/RS analysis
110. Hybrid - Temporal Fusion Transformers - Combines feature-wise transformations with temporal attention; multi-horizon EO/RS forecasting
111. Hybrid - BIT (Bi-Temporal Transformer with CNN) - Combines CNN feature extraction with transformer temporal reasoning for change detection
112. Hybrid - ChangeFormer - Transformer-CNN hybrid specifically optimized for EO/RS change detection
113. Hybrid - STANet (Spatial-Temporal Attention) - Hybrid attention mechanisms for spatio-temporal change analysis
114. Hybrid - DeepLab-Transformer Hybrids - Atrous spatial pyramid pooling with self-attention; multi-scale EO/RS segmentation
115. Hybrid - HRNet-Transformer - High-resolution networks enhanced with attention; precise boundary detection
116. Hybrid - Mask2Former - Unified segmentation framework combining transformers and mask classification
117. Hybrid - EfficientFormer - Transformer-CNN hybrid optimized for mobile devices; real-time EO/RS processing
118. Hybrid - PoolFormer - Replaces attention with simple pooling; efficient alternative for large-scale EO/RS mapping
119. Hybrid - EdgeNeXt - CNN-Transformer hybrid for edge computing; suitable for onboard satellite processing
120. Hybrid - Diffusion-Transformer Hybrids - Combines diffusion probabilistic models with transformer conditioning; high-quality EO/RS image generation
121. Hybrid - GAN-Transformer Networks - Transformer-based discriminators with CNN generators; improved EO/RS synthetic data quality


8. Emerging Architectures.
122. Emerging - Retentive Networks as SSM alternatives
123. Emerging - Neural Radiance Fields (NeRF) for 3D reconstruction
124. Emerging - Foundation Models for EO/RS - Future paradigm shift