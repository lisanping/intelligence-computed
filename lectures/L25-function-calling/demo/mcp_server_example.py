"""
第 25 讲 · 最小 MCP Server 示例
==================================

演示：
  1. 用 Python MCP SDK 构建一个最小 MCP Server
  2. 暴露三个 Tools：天气查询、文件读取、文件写入
  3. 展示 MCP Server 的标准生命周期（initialize → list_tools → call_tool）

依赖：pip install mcp httpx
运行方式：
  # 直接运行查看 Server 信息
  python mcp_server_example.py

  # 作为 MCP Server 被 Client 调用（stdio 传输）
  # 在 MCP Client 配置中添加:
  # {
  #   "mcpServers": {
  #     "l19-demo": {
  #       "command": "python",
  #       "args": ["mcp_server_example.py", "--serve"]
  #     }
  #   }
  # }

种子：所有实验使用 seed=1337

注意：此文件是教学演示，展示 MCP Server 的核心结构。
生产环境需要额外的安全措施（认证、授权、输入验证、速率限制）。
"""

import argparse
import json
import os
import sys
from typing import Any


# ──────────────────────────────────────────────
# 0. 参数解析
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCP Server example for L31")
    p.add_argument("--serve", action="store_true",
                   help="Run as MCP Server (stdio transport)")
    p.add_argument("--info", action="store_true",
                   help="Print server info and exit")
    return p.parse_args()


# ──────────────────────────────────────────────
# 1. 工具实现
# ──────────────────────────────────────────────
MOCK_WEATHER: dict[str, dict[str, Any]] = {
    "北京": {"temp": 22, "condition": "晴", "humidity": 45, "wind": "北风3级"},
    "上海": {"temp": 26, "condition": "多云", "humidity": 72, "wind": "东南风2级"},
    "东京": {"temp": 18, "condition": "小雨", "humidity": 85, "wind": "东风4级"},
    "纽约": {"temp": 15, "condition": "阴", "humidity": 60, "wind": "西风3级"},
}

# 限制文件操作的工作目录
ALLOWED_DIR = os.path.abspath(os.path.dirname(__file__))


def get_weather(city: str, unit: str = "celsius") -> dict[str, Any]:
    """获取指定城市天气（模拟数据）。"""
    weather = MOCK_WEATHER.get(city)
    if weather is None:
        return {"error": f"城市 '{city}' 无法识别。支持: {', '.join(MOCK_WEATHER.keys())}"}
    result = {**weather}
    if unit == "fahrenheit":
        result["temp"] = round(result["temp"] * 9 / 5 + 32, 1)
        result["unit"] = "°F"
    else:
        result["unit"] = "°C"
    return result


def read_file(path: str) -> dict[str, Any]:
    """读取文件内容（限制在工作目录内）。"""
    abs_path = os.path.abspath(os.path.join(ALLOWED_DIR, path))
    if not abs_path.startswith(ALLOWED_DIR):
        return {"error": "安全限制：只能读取工作目录下的文件"}
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read(10_000)
        return {"content": content, "path": path, "size": len(content)}
    except FileNotFoundError:
        return {"error": f"文件 '{path}' 不存在"}
    except OSError as e:
        return {"error": str(e)}


def write_file(path: str, content: str) -> dict[str, Any]:
    """写入文件内容（限制在工作目录内）。"""
    abs_path = os.path.abspath(os.path.join(ALLOWED_DIR, path))
    if not abs_path.startswith(ALLOWED_DIR):
        return {"error": "安全限制：只能写入工作目录下的文件"}
    try:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"success": True, "path": path, "bytes_written": len(content.encode("utf-8"))}
    except OSError as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────
# 2. MCP Server 定义
# ──────────────────────────────────────────────
# MCP Server 的核心结构：
#   - server info（名称、版本）
#   - tools（暴露的工具列表 + 每个工具的 handler）

SERVER_NAME = "l19-demo-server"
SERVER_VERSION = "0.1.0"

# 工具描述（MCP 格式）
TOOL_DEFINITIONS = [
    {
        "name": "get_weather",
        "description": "获取指定城市的当前天气信息。当用户询问天气、气温、是否需要带伞时使用。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如'北京'、'上海'、'Tokyo'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位，默认 celsius"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "read_file",
        "description": "读取指定文件的全部内容。只能读取 MCP Server 工作目录下的文件。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径（相对于 Server 工作目录）"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "将内容写入指定文件。只能写入 MCP Server 工作目录下的文件。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径（相对于 Server 工作目录）"
                },
                "content": {
                    "type": "string",
                    "description": "要写入的文本内容"
                }
            },
            "required": ["path", "content"]
        }
    },
]


def handle_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """路由工具调用到对应的实现函数。"""
    if name == "get_weather":
        result = get_weather(
            city=arguments["city"],
            unit=arguments.get("unit", "celsius"),
        )
    elif name == "read_file":
        result = read_file(path=arguments["path"])
    elif name == "write_file":
        result = write_file(
            path=arguments["path"],
            content=arguments["content"],
        )
    else:
        result = {"error": f"未知工具: {name}"}

    return json.dumps(result, ensure_ascii=False)


# ──────────────────────────────────────────────
# 3. MCP Server 运行（使用 MCP Python SDK）
# ──────────────────────────────────────────────
def run_mcp_server() -> None:
    """
    使用 MCP Python SDK 启动 Server（stdio 传输）。

    等价于以下 MCP SDK 代码：

        from mcp.server import Server
        from mcp.server.stdio import stdio_server

        server = Server(SERVER_NAME)

        @server.list_tools()
        async def list_tools():
            return TOOL_DEFINITIONS

        @server.call_tool()
        async def call_tool(name, arguments):
            result = handle_tool_call(name, arguments)
            return [TextContent(type="text", text=result)]

        async def main():
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
        import asyncio

        server = Server(SERVER_NAME)

        @server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name=t["name"],
                    description=t["description"],
                    inputSchema=t["inputSchema"],
                )
                for t in TOOL_DEFINITIONS
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            result = handle_tool_call(name, arguments)
            return [TextContent(type="text", text=result)]

        async def main() -> None:
            async with stdio_server() as (read, write):
                await server.run(
                    read, write,
                    server.create_initialization_options(),
                )

        asyncio.run(main())

    except ImportError:
        print("❌ MCP SDK 未安装。请运行: pip install mcp")
        print("\n以下是 Server 的结构展示（无需 SDK）：")
        print_server_info()
        sys.exit(1)


# ──────────────────────────────────────────────
# 4. Server 信息展示（教学用）
# ──────────────────────────────────────────────
def print_server_info() -> None:
    """打印 Server 信息和工具列表（教学展示）。"""
    print("=" * 60)
    print(f"MCP Server: {SERVER_NAME} v{SERVER_VERSION}")
    print("=" * 60)

    print(f"\n📋 暴露 {len(TOOL_DEFINITIONS)} 个工具:")
    for i, tool in enumerate(TOOL_DEFINITIONS, 1):
        print(f"\n  {i}. 🔧 {tool['name']}")
        print(f"     描述: {tool['description']}")
        props = tool["inputSchema"].get("properties", {})
        required = tool["inputSchema"].get("required", [])
        for pname, pdef in props.items():
            req = " (必需)" if pname in required else " (可选)"
            print(f"     参数: {pname}: {pdef['type']}{req} — {pdef.get('description', '')}")

    print("\n" + "-" * 60)
    print("📡 传输方式: stdio (标准输入/输出)")
    print(f"📁 工作目录: {ALLOWED_DIR}")

    print("\n🔗 Client 配置示例:")
    config = {
        "mcpServers": {
            "l19-demo": {
                "command": "python",
                "args": [os.path.basename(__file__), "--serve"]
            }
        }
    }
    print(json.dumps(config, indent=2, ensure_ascii=False))

    # 演示工具调用
    print("\n" + "-" * 60)
    print("🧪 工具调用演示:")

    test_calls = [
        ("get_weather", {"city": "北京"}),
        ("get_weather", {"city": "东京", "unit": "fahrenheit"}),
        ("get_weather", {"city": "拉萨"}),  # 不存在的城市
        ("read_file", {"path": "nonexistent.txt"}),  # 不存在的文件
    ]

    for name, args in test_calls:
        result = handle_tool_call(name, args)
        print(f"\n  调用: {name}({json.dumps(args, ensure_ascii=False)})")
        print(f"  结果: {result}")

    # 对比 Function Calling 直连 vs MCP
    print("\n" + "=" * 60)
    print("📊 Function Calling 直连 vs MCP 解耦:")
    print("=" * 60)
    print("""
  ┌───────────────────────┬──────────────────────────┬──────────────────────────┐
  │        维度           │  Function Calling 直连   │      MCP 解耦            │
  ├───────────────────────┼──────────────────────────┼──────────────────────────┤
  │ 工具定义位置           │ 在你的应用代码中          │ 在 MCP Server 的          │
  │                       │                          │ list_tools() 中           │
  ├───────────────────────┼──────────────────────────┼──────────────────────────┤
  │ 工具执行位置           │ 在你的应用代码中          │ 在 MCP Server 的          │
  │                       │                          │ call_tool() 中            │
  ├───────────────────────┼──────────────────────────┼──────────────────────────┤
  │ 换模型影响             │ 可能需要改工具定义格式    │ Server 不用改             │
  ├───────────────────────┼──────────────────────────┼──────────────────────────┤
  │ 工具升级影响           │ 改应用代码 → 重新部署     │ 改 Server → 应用不用改    │
  ├───────────────────────┼──────────────────────────┼──────────────────────────┤
  │ 加新工具              │ 改代码 + 改工具列表       │ 部署新 Server + 加配置行  │
  ├───────────────────────┼──────────────────────────┼──────────────────────────┤
  │ 复用性                │ 每个应用各自实现          │ 一个 Server 供所有应用用  │
  └───────────────────────┴──────────────────────────┴──────────────────────────┘

  💡 MCP 的核心价值: 解耦 → 一次实现，处处可用。
  """)


# ──────────────────────────────────────────────
# 5. 主函数
# ──────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if args.serve:
        # 作为 MCP Server 运行（stdio 传输）
        run_mcp_server()
    else:
        # 打印 Server 信息（教学模式）
        print_server_info()


if __name__ == "__main__":
    main()