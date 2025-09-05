from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility

# 配置基本信息
YOUR_ACCESS_TOKEN = "ms-9fxx7e"  # 填写自己的 api token
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

# 取名字
owner_name = "xx"  # ModelScope 的用户名，需根据自己情况修改
model_name = "xx"  # 为模型库取个响亮优雅又好听的名字，需根据自己情况修改
model_id = f"{owner_name}/{model_name}"

# 创建模型仓库
api.create_model(
    model_id,
    visibility=ModelVisibility.PUBLIC,
    license=Licenses.APACHE_V2,
    chinese_name=f"{owner_name}的 latex 分类",
)

# 上传模型到仓库
api.upload_folder(
    repo_id=f"{owner_name}/{model_name}",
    folder_path="xx",  # 微调后模型的文件夹名称
    commit_message="upload model folder to repo",  # 写上传信息
)
