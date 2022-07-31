# coding:utf-8

team_names = [
            # "徐州医科大学附属医院骨科", 
            #   "中国首都医科大学宣武医院", 
            #   "京大学心理与认知科学学院及麦戈文脑研究所韩世辉教授课题组",
            #   "瑞金医院",
            #   "南京医科大学第一附属医院神经内科",
            #   "北京航空航天大学生物与医学工程学院",
            #   "武汉大学中南医院神经内科",
            #   "济宁医学院附属医院医学影像科",
            #   "福建医科大学附属协和医院乳腺外科",
            #   "广东省佛山市第一人民医院核医学科",
            #   "大连医科大学附属第一医院神经内科",
            #   "中国科学院深圳先进技术研究院研究员郑炜团队",
            #   "中国科学院深圳先进技术研究院研究员郑炜团队",
              "华中科技大学武汉光电国家实验室（筹）生物医学光子研究中心",
              ]
years = [
        # "2022", 
        #  "2022", 
        #  "2021", 
        #  "2019", 
        #  "2022", 
        #  "2022", 
        #  "2021", 
        #  "2022", 
        #  "2022", 
        #  "2021", 
         
        #  "2022", 
        #  "2020", 
        #  "2020", 
         "2021"]
journal_names = [
                # "Cellular signalling", 
                #  "European radiology", 
                #  "NeuroImage", 
                #  "International Journal of Neuroscience",
                #  "中华行为医学与脑科学杂志",
                #  "磁共振成像",
                #  "Annals of translational medicine",
                #  "医学影像学杂志",
                #  "中华核医学与分子影像杂志",
                #  "国际放射医学核医学杂志",
                #  "Cerebral Cortex",
                #  "Biomedical Optics Express",
                #  "Optics Express"
                 "Nature Methods",
                 ]
paper_en_names = [
                # "IL-10 inhibits osteoclast differentiation and osteolysis through MEG3/IRF8 pathway", 
                #   "[18F]FDG PET/MRI and magnetoencephalography may improve presurgical localization of temporal lobe epilepsy",
                #   "Neural dynamics of pain expression processing: Alpha-band synchronization to same-race pain but desynchronization to other-race pain",
                #   "Utility of stereo-electroencephalography recording guided by magnetoencephalography in the surgical treatment of epilepsy patients with negative magnetic resonance imaging results",
                #   "Resting-state fMRI study of Parkinson disease patients with peak-dose dyskinesia: an ALFF analysis",
                #   "Identification of Alzheimer's disease and mild cognitive impairment patients using individual-specific functional connectivity",
                #   "Brain function state in different phases and its relationship with clinical symptoms of migraine: an fMRI study based on regional homogeneity (ReHo)",
                #   "Characteristis of 18F-FDG PET/CT in gastric carcinomas before treatment",
                #   "The predictive value of 99TCM-3PRGD2 SPECT imaging for pathological complete response after neoadjuvant chemotherapy in breast cancer and comparison with 18F-FDG PET/CT",
                #   "Advances of 18F-FDG PET/CT in PD-1/PD-L1 targeted immumotherapy of tumors",
                #   "Greater prefrontal activation during sitting toe tapping predicts severer freezing of gait in Parkinson's disease: an fNIRS study",
                #   "Axial resolution improvement of two-photon microscopy by multi-frame reconstruction and adaptive optics",
                #   "Adaptive optics via pupil ring segmentation improves spherical aberration correction for two-photon imaging of optically cleared tissues",
                  "High-definition imaging using line-illumination modulation microscopy",
                  ]
paper_zh_names = [
                # "IL-10通过MEG3/IRF8通路抑制破骨细胞分化和骨溶解",
                #   "[18F]FDG PET/MRI和脑磁图可改善颞叶癫痫的术前定位",
                #   "疼痛表达加工的神经动力学:阿尔法带对同种族疼痛的同步，而对其他种族疼痛的去同步",
                #   "脑磁图引导下立体脑电图记录在mri阴性癫痫患者手术治疗中的应用",
                #   "帕金森病合并剂峰异动患者低频振幅的静息态fMRI研究",
                #   "基于个体特异性功能连接的阿尔茨海默病早期识别研究",
                #   "偏头痛不同阶段的脑功能状态及其与临床症状的关系：基于Reho的fMRI研究",
                #   "治疗前胃癌患者18F-FDG PET/CT显像特征分析",
                #   "99Tcm-3PRGD2SPECT显像对乳腺癌新辅助化疗后病理完全缓解的预测价值及与18F-FDG PET/CT的对比研究",
                #   "18F-FDG PET/CT在肿瘤PD-1/PD-L1免疫治疗中的研究进展",
                #   "一项近红外光谱研究显示，坐着时轻叩脚趾时前额叶更大的激活预示着帕金森病中更严重的步态冻结",
                #   "采用多帧重建和自适应光学技术提高双光子显微镜的轴向分辨率",
                #   "通过瞳孔环分割的自适应光学改善了光学清除组织双光子成像的球差校正",
                  "用线照明调制显微术实现高清成像",
                  ]
research_contents = [
                    # "磨损颗粒引起的骨溶解是关节置换术失败的主要原因，抑制破骨细胞分化可减轻磨损颗粒诱导的骨溶解。本研究旨在探讨lncRNA母系表达基因3 (MEG3)对破骨细胞分化及磨损颗粒诱导的骨溶解的影响，并完善白细胞介素-10 (IL-10)抑制破骨细胞分化的潜在机制。结论MEG3通过与STAT1结合调控IRF8的表达，从而影响破骨细胞分化和磨损颗粒诱导的骨溶解。IL-10可能抑制MEG3/IRF8介导的破骨细胞分化。",
                    # "团队通过73例定位相关TLE患者接受了[18F]FDG PET/MRI和MEG检查。PET/MRI图像由两名放射科医生判读；使用统计参数映射(SPM)确定PET的局灶性代谢低下。MEG信号源与t1加权序列共配准，用Neuromag软件进行分析。以皮质切除和手术结果为标准评估[18F]FDG PET/MRI、MEG和PET/MRI/MEG定位EZ的临床价值。分析手术结果与符合或不符合皮质切除的手术方式之间的相关性。术前通过[18F]FDG PET/MRI/MEG评估可以提高对TLE EZ的识别，进一步指导手术决策。",
                    # "通过表情理解他人的情感状态是人脑的基本功能，但他人的社会属性（比如种族）影响我们理解并分享他人的情感（即共情）。该研究在中国被试记录了判断亚裔和白人面孔表情（中性表情或疼痛表情）任务中的脑磁图。通过分析和溯源不同频率神经震荡活动发现，相比于中性亚裔面孔，疼痛亚裔面孔诱发alpha活动的同步增强，低和高alpha活动分别于刺激呈现后130ms左右在颞顶联合区、100毫秒左右在左侧脑岛开始。然而，相比于中性白人面孔，疼痛白人面孔诱发alpha活动的同步活动减弱，低和高alpha活动减弱分别于刺激呈现后60ms左右在感觉运动皮层、40毫秒左右在左侧脑岛开始。并且，左侧感觉运动皮层和左侧脑岛在观看疼痛表情时的功能连接可以预测受试者对他人情感的主观感受。这些发现提示观察者与被观察者之间的种族关系可能在不同时程调控共情的感觉运动、情感和认知成分，产生对本族和他族面孔疼痛表情不同的动态神经活动。该论文报道的实验发现对于理解种族关系调控痛觉动态共情神经活动提供了神经科学依据。",
                    # "对于神经外科医生来说，给没有检测到结构性病变的患者进行手术非常有挑战性。 因此，这项回顾性研究旨在探讨在MRI阴性的癫痫患者中以脑磁图（MEG）-磁共振成像（MRI）重建为指导，在可疑区域中做立体脑电图（SEEG）的手术结果。本研究包括47例MRI阴性的癫痫患者。22（47％）位患者达到了完全无癫痫发作的状态。 性别，习惯，年龄和病程与结局无癫痫发作的预后没有显著相关性（p=0.187 [Pearson卡方检验]）。 随访时，预后良好的患者（Engle I和II）高达68％。 此外，在SEEG和MEG一致组中发现了更多的无癫痫发作的患者。SEEG是评估切除性癫痫手术，特别是阴性MRI癫痫患者的有价值的工具。MEG极大地促进了SEEG电极植入的定位。 但是，这些工具都不是绝对敏感和可靠的。 因此，在癫痫手术中收集尽可能多的信息以获得令患者和医生满意的结果是非常有必要的。",
                    # "团队观察帕金森病(Parkinson's disease,PD)合并剂峰异动患者的静息态脑活动特点,并探索其发病机制；并得出右侧额下回和右侧辅助运动区脑自发活动异常可能是PD患者出现剂峰异动的神经生物学基础。右侧额下回脑活动异常与剂峰异动患者的病情严重程度相关，其ALFF值是识别剂峰异动患者潜在的影像标志物的结论。",
                    # "团队基于静息态功能磁共振成像探索个体特异性功能连接对阿尔茨海默病及轻度认知障碍患者,稳定型轻度认知障碍及进展型轻度认知障碍患者分类的影响；得出采用蕴含更多个体特性的个体特异性连接可提高对AD及MCI识别准确度，个体特异性功能连接有望作为AD及MCI诊断的潜在神经影像学标志物的结论。",
                    # "团队通过区域同质性（ReHo）分析偏头痛患者不同阶段脑区的激活情况，探讨其与临床症状的关系。我们需要从整体上观察偏头痛的病程。即使在发作间期，也可能通过楔叶和舌回影响疾病的发展。ACC通过诱导抗损伤感觉调节功能来调节偏头痛的不同状态。旁中心小叶不仅与偏头痛发作有关，而且与频率有关。它可能对随后的偏头痛的结果产生影响，无论是慢性偏头痛，还是大脑的重塑。ACC通过诱导抗损伤感觉调节功能来调节偏头痛的不同状态。旁中心小叶不仅与偏头痛发作有关，而且与频率有关。它可能对随后的偏头痛的结果产生影响，无论是慢性偏头痛，还是大脑的重塑。ACC通过诱导抗损伤感觉调节功能来调节偏头痛的不同状态。旁中心小叶不仅与偏头痛发作有关，而且与频率有关。它可能对随后的偏头痛的结果产生影响，无论是慢性偏头痛，还是大脑的重塑。",
                    # "论文探讨治疗前胃癌患者18F-FDG PET/CT显像特征，并分析影响胃癌原发灶最大标准摄取值（maximum standardized uptake value, SUVmax）的相关因素。 PET/CT检查在胃癌原发灶诊断及淋巴结、脏器转移探查中具有重要价值，病理分型、脏器转移是影响原发灶SUVmax值的相关因素。",
                    # "论文探讨99Tcm-联肼尼克酰胺-3聚乙二醇-精氨酸-甘氨酸-天冬氨酸环肽二聚体（3PRGD2）显像对乳腺癌患者新辅助化疗（NAC）病理完全缓解（pCR）的预测价值, 并将其与18F-FDG显像比较。最后得出乳腺癌患者在NAC后, 乳腺癌原发灶和ALN转移灶对99Tcm-3PRGD2摄取的早期变化可用于预测pCR, 其中ALN转移灶99Tcm-3PRGD2显像的早期变化对pCR的预测效能可能较18F-FDG显像更高。",
                    # "程序性死亡受体1（PD-1）/程序性死亡配体1（PD-L1）信号通路产生负性调控信号介导肿瘤免疫逃逸,导致肿瘤免疫耐受,促进其进展。而PD-1/PD-L1免疫治疗可恢复肿瘤微环境的免疫反应,介导T细胞增殖、活化,杀伤相关肿瘤细胞,为恶性肿瘤的治疗提供新方法。作者就18F-FDG PET/CT在肿瘤PD-1/PD-L1免疫治疗中的研究进展进行综述。",
                    # "先前的研究表明，与没有步态冻结(FoG)的帕金森病(PD)患者相比，步态冻结患者在进行下肢运动(包括站立、行走和转弯)时表现出更大的前额叶激活，这些运动需要运动控制和平衡控制。然而，FoG与纯运动控制的关系及其潜在机制尚不清楚。结果表明，伴有脑雾的PD患者需要额外的认知资源来补偿其受损的运动控制自动性，这在重度脑雾患者中比轻度脑雾患者更明显。",
                    # "双光子显微技术以其深层穿透和天然层析能力在生物成像中发挥重要作用，尤其是在大脑神经环路成像中。然而，传统双光子成像技术的轴向分辨率一般是几个微米甚至更差，大于亚微米尺度的横向分辨率，不利于分辨三维分布的一些精细结构，如神经环路上沿轴向分布的轴突、树突和突触等结构。该研究中，研究人员将多帧重构算法用于双光子成像，结合自主研发的自适应光学模块和相位补偿方法，实现双光子成像轴向分辨率3倍提升，信噪比提升超过3倍。利用该系统，研究人员对小鼠离体脑片和活体大脑进行成像研究，观测到一般双光子成像无法分辨的轴向细节，包括胞体的精细连接、更加清晰的轴突边界和小胶质突起等；实时追踪败血症小鼠模型，清晰观察到小胶质细胞的三维形态变化。该研究提供了一种三维高分辨成像技术，有利于进一步了解脑机理和诊治重大脑疾病。",
                    # "双光子显微结合组织光透明技术能够在样品深层处进行亚微米级的荧光成像，这对研究神经环路、连接和功能具有重要意义。但是，组织光透明剂处理后的样品与物镜的标准浸润介质的折射率不匹配，这会引入球差、降低双光子的成像分辨率和荧光信号的强度。针对该问题，研究人员提出新型环形矫正的自适应光学方法来补偿折射率不匹配，从而降低球差提高成像质量。",
                    "团队提出了一种高清晰度、高通量的光学层析显微成像新方法——线照明调制光学层析成像（Line-illumination modulation microscopy, LiMo），巧妙地将线照明光强的高斯分布作为一种天然的照明强度调制模式，不同照明强度对焦面上的信号产生相应调制，而对焦外背景信号不调制。该方法克服了传统结构光照明成像中存在残留调制伪影的固有缺陷，也无需多次成像即可获得所需数据，并具有线扫描对大范围样本成像通量高的优点，解决了传统荧光显微光学层析成像方法无法同时兼顾高分辨率、高通量和高清晰度的问题。"]
temp = ""
for team_name, year, journal_name, paper_en_name, paper_zh_name, research_content in zip(team_names, years, journal_names, paper_en_names, paper_zh_names, research_contents):
    temp += f"{team_name}团队于{year}年在《{journal_name}》期刊上发表“{paper_en_name}”（{paper_zh_name}）。{research_content}\n\n"
    
print(temp)