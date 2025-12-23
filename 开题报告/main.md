
<h1>论文</h1>
<h2>CAD-Assistant</h2>
[[CAD-Assistant.pdf]]
- **做了什么**：把LLM当“规划器”，通过提示词+函数文档去调用 **FreeCAD Python API**，用多轮“PLAN→写代码→执行→看结果→修正”完成建模/草图约束等任务；实现形态更接近“CAD Copilot/Agent”。

- **硬件要求**：论文本身重点在**系统与工具链**，不强调本地大规模训练；你主要需要能跑 FreeCAD + Python，以及可用的LLM（本地或API）

- **对你最有用的点**：这就是你 CAD-Agent 的**Agent骨架**（工具接口、动作空间、可执行反馈回路），非常适合作为 phase0 直接落地。
<h2>CAD-Coder</h2>
- **做了什么**：构造/使用带“设计流程代码”的数据（如 CADQuery/脚本），让模型学会生成可执行的 CAD 代码；数据生成阶段用了多卡，但微调阶段非常轻。

- **硬件（论文报告）**：数据生成用了 **6×V100**，微调只用 **单张 V100**。

- **4090适配建议**：你只有 1–2×4090 时，这篇是**最接近“能复现且能改成你实验室标准”** 的路线之一：把“CADQuery脚本”换成 **FreeCAD宏/FreeCAD Python建模脚本**，做成你们轴类零件的指令数据，就能直接SFT/LoRA。

<h2>FLEXCAD(ICLR2025)</h2>
- **做了什么**：把 CAD 序列转成文本，让 LLM 学会在**不同层级**（草图/拉伸等）做**可控生成/局部补全**；推理时通过 mask/infill 做“局部编辑/补全”

- **硬件（论文报告）**：Llama-3-8B 用 LoRA（只训 0.042% 参数），训练在 **4×A6000** 上

- **4090适配建议**：你可以用 **1–2×4090 做“缩配版FlexCAD”**：
	- 选 7B/8B base（Llama3/Qwen2.5），QLoRA 4bit + 梯度检查点；
	- 数据先从你用 FreeCAD 程序化生成的“轴类”开始（更短序列、更规整），再逐步引入复杂件

-  **对你最有用的点**：它提供了“**编辑=mask一段→infill→再执行验证**”的范式，非常适合你后续做 CAD-Editor 类能力的工程底座

<h2>GenCAD</h2>
- **做了什么**：围绕 CAD 命令序列做更强的表示学习（自回归encoder-decoder），并扩展到 **image→CAD latent / retrieval / generation** 等

- **硬件（论文报告）**：训练在 **单张 80GB A100** 上完成

- **4090适配建议**：可以做，但要“缩模型/缩batch/缩序列长度”，更现实的做法是：
	- 用它的思想（**CAD序列→latent→检索/条件生成**）做轻量版本
	- 或者把 GenCAD 当作 phase2 的“**检索模块/表示学习模块**”，不必完整复现所有分支任务

- **数据关键点**：它依赖 DeepCAD 这种“带设计历史的序列数据”，而你只有 STEP 时需要额外解决“标签从哪来”。

<h2>Text2CAD</h2>
[[Text2CAD]]
- **做了什么**：提出从“抽象→入门→中级→专家”多层级文本提示生成 CAD 序列，并强调用更细粒度文本提高生成质量

- **硬件（论文报告）**：训练 **单张 A100 80GB，约2天**。

- **工程复杂度提醒**：它的数据标注流水线会用到额外的 VLM/LLM 来生成描述与“最小元数据”等，属于“数据工程很重”的路线。

- **4090适配建议**：你可以只复现核心：**文本→序列模型 + FreeCAD执行验证**；但“完整复现论文的数据标注链”会耗时

<h2>CAD-Llama(数据工程重)</h2>
- **做了什么**：强调通过结构化/自生成数据（如 SPCC 等）提升 LLM 在 CAD 任务上的能力，并进行自适应预训练与指令微调。

- **硬件/成本（论文报告）**：预训练与微调用了 **4×A100**；并报告了 A100 GPU-hours 与 GPT-4o token 级别的成本（例如 38/12 A100 GPU-hours、约 1.66M prompt tokens 等）。

- **4090适配建议**：如果你不走“重预训练”，只做 **LoRA/QLoRA 指令微调**，并把数据限定在“轴类零件 + FreeCAD可执行脚本”，是可控的；但它的核心难点在**高质量指令数据构造**而不在训练本身。

<h2>CAD-Editor(训练吃显存但是值得借鉴)</h2>
[[CAD-Assistant.pdf]]
- **做了什么**：专门研究 **text-based CAD editing**，用 **locate-then-infill** 把“编辑位置定位”和“内容生成”解耦；并构造约 12 万的合成编辑数据
- **硬件（论文报告）**：用 GPT-4o 做视觉理解，序列模型用 Llama-3-70B；训练在 **4×A800-80GB** 上
- **4090适配建议**：**不建议原尺寸原配置硬复现**；但你完全可以在 phase2 做“缩配版CAD-Editor”：
	- base 换 7B/8B；
	- 用 FlexCAD 的 mask/infill 思路 + FreeCAD 执行验证回路；
	- 把“locate”先做成规则/轻模型（例如几何特征检索），把“infill”交给LLM。

<h2>CAD RL / cadrille(8张H00 RL)</h2>
- **做了什么**：指出仅靠SFT会受“**程序不一致/多解**”影响，于是引入 **RL（PPO）**，让模型通过奖励信号（几何一致性等）去学更稳的 CAD 生成
- **硬件（论文报告）**：SFT 用 **单张 H100**，RL 用 **8×H100**
- **4090适配建议**：这篇**不适合完整复现RL规模**；但它给了你一个非常关键的“只有STEP也能往前走”的思路：**没有标注程序时，用几何reward做自我改进**（你可以做轻量版：SFT + 小步RL/拒绝采样 + 几何验证）

<h2>DeepCAD(经典数据表示基线)</h2>
[[DeepCAD]]  [[2105.09492v2.pdf]]

- **做了什么**：把 CAD 表示成“草图+拉伸”等命令序列，并给出深度生成模型与数据集；同时也明确指出 fillet 这类操作需要引用B-rep，较难纳入序列模型。
- **对你项目的意义**：你要做“能训练的CAD大模型”，必须先选一种**可学习的表示**；DeepCAD/其后续（FlexCAD/GenCAD/Text2CAD）都是围绕这个表示体系在做


<h1>研究路线</h1>

<h2>一：利用FreeCAD API  补全</h2>
<h2>数据条件</h2>
**只有STEP**
STEP只有B-rep几何，不包含草图，约束，特征树，因此无法进行监督学习
1. **自选可监督数据** FreeCAD 程序化生成零件 脚本-STEP成对，再用STEP做适配
2. **弱监督/自监督/RL**  用“几何相似度”当reward， 把STEP 当 target，让模型自我修正/搜索/修正 
3. 
**原生文件**
Free CAD `.FCStd` Fusion360 `.f3d` Onshape FeatureScript CADQuery 

用户特征 -> 特征序列 Json 

<h2>Phase0</h2>
<h3>FreeCAD+数据工程 构建训练集</h3>
不训练的情况下跑通 任务->脚本->执行->几何验证->修正


<h4>0.1 环境与接口 </h4>
 - [ ] FreeCAD headless 运行 与 Python API 调用封装 
 - [ ] 统一输出: `FCStd+STEP+mesh`, 渲染多视图图像
 - [ ] 设计工具函数：创建平面草图，拉伸旋转

<h4>0.2 条件下的数据构造 </h4>
- [ ] 程序化生成零件家族 用FreeCAD Python 随机采样参数 生成可编辑脚本同时导出 STEP
- [ ] 自动生成结构化规格说明（JSON) 与自然语言说明(Prompt)
- [ ] 得到`text/spec - FreeCAD脚本 - STEP` 作为训练集

<h4>0.3测评与可视化</h4>
- [ ] 几何一致性 
- [ ] 可编辑性指标：脚本是否可以重放，参数是否可控，特征是否单独修改

<h4>0.4 无训练 Agent baseline</h4>
- [ ] 直接通用LLM+CAD-Assistant 的提示模式，让他学会调用工具函数建模


<h2>Phase 1  领域CAD大模型(SFT/LoRA)</h2>

>目标：训练出模具领域的CAD脚本生成模型，并用FreeCAD执行为硬约束评测

<h4>1.1 模型选择 </h4>

 - 7B/8B 级别开源LLM(QLoRA/LoRA)
 - 方法论：CAD-Coder 的 “指令” -->可执行的代码

<h4>1.2训练数据组织</h4>
- 核心监督：`prompt/结构化spec -> FreeCAD Python脚本`
- 辅助监督:`STEP点云渲染/多视图->SPEC`
- 数据难度课程：纯旋转体/台阶轴 -> 孔槽 -> 倒角 -> 布尔组合

<h4>训练与约束</h4>
- [ ] SFT:要求脚本能执行：把执行报错最为重采样信号
- [ ] 输出格式约束：只允许调用你封装的FreeCAD 工具函数
- [ ] 自动单元测试：每个脚本都执行一遍，失败样品回流到负样品集


论文复现：CAD-Coder/ FlEXCAD

<h2>Phase2 :CAD-Agent 智能体化+自动修正 </h2>
> 能规划，能编辑，能自我修复 

<h4>2.1 Agent工作流</h4>
1. 解析需求 -> 结构化 spec
2. 规划建模步骤 
3. 生成脚本-> FreeCAD
4. 验证：几何指标+关键尺寸检查
5. 失败：定位问题 -->局部编辑 -->重试

<h4>2.2利用STEP文件</h4>
- [ ] 把真实STEP转换成: 多视图渲染/点云/B-rep 特征图输入
- [ ] 用reward进行改进：拒绝采样/多候选生成，选几何接近者

<h4>2.3 加入编辑能力</h4>
- [ ] CAD-Editor：先locate(找要改的特征/面/草图),再infill(改对应的脚本/参数)
- [ ] 定义模具类的locate


参考论文：CAD- Editor/CAD RL 

<h2>利用已经复现的论文 Text2CAD 精简数据集进行 微调</h2>

<h3>Text2CAD的样本单位与筛选入口</h3>
1. Text2CAD每个CAD模型对应四种提示词等级(L0,L1,L2,L3:abstract/beginner/intermediate/expert),论文里评估CAD通常指用L3
2. 同时它在数据预处理会把DeepCAD原始JSON改成minimal metadata(去掉随机KEY和冗余字段)
3. 提示词多样性：同一个形状可能不出现明确物体名，不能只靠"关键词=物体名"来筛选

因此：
**筛选最好以L3文本为主，几何/序列特征为辅助**
**切分必须以CAD model id 为单位**

<h3>总体步骤</h3>

<h4>step1:拉取并盘点数据结构</h3>
- model_CSV
	- model_id 
	- prompt_L0,L1,L2
- cad_sequence 
- (op) minimal metadata
<h4>Step2:定义模具领域的筛选规则</h4>
文本判别：关键词/语义相似度(embedding)/小分类起 
CAD序列：从minimal json/metadata中提取

<h4>筛选出统计报告,用model_id切分train/val/test</h4>
- 以model_id 分组后在split
- 评估集用L3 prompt ，训练可混合L1，L2，但L3权重更重

<h4>Step3 导出数据集对模型进行微调</h4>

<h4> Step4 构建智能体</h4>
- Tool1:`genereate_cad(prompt) -> cad_sequence`
- Tool2: 用现有脚本重建渲染
- Agent Loop: 学习错误日志

<h3>WHUCAD 包含比Text2CAD更多的特征</h3>
1. **拿到数据 & 跑通解析**
	-  用仓库里的 loader 把 `.h5` 解成命令序列 + 参数
2.  统计 WHUCAD 的命令集合 `C_whucad`,统计当前 Text2CAD/DeepCAD 的命令集合`C_text2cad`分成两堆：`C_whucad ∩ C_text2cad`（可直接用）和 `C_whucad \ C_text2cad`（要丢/要扩展）
3. **做数据切分（**  
- **规则/几何启发式**：比如只保留某些尺寸范围、特征数量范围、是否含孔/倒角等（
- **VLM 打标**：用多视图渲染图让 VLM 给类别/用途/部件类型标签，再按标签筛（[GitHub](https://github.com/fazhihe/WHUCAD)）
- **LLM 读历史生成标签**：把命令序列摘要喂给 LLM 生成部件类型/工艺特征的标签
1. **补齐 Text2CAD 需要的文本 
	先从 CAD 序列抽一个 **minimal metadata**再生成 L0/L1/L2/L3（文本
2. **组装成训练对： (text prompt, CAD sequence)**  
    把 WHUCAD 的样本格式对齐到已经复现的 Text2CAD dataloader。
3. **LoRA 微调 + 评测**
