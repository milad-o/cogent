"""Built-in task handlers for common SSIS task types."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

from agenticflow.capabilities.ssis.handlers.base import TaskHandler

if TYPE_CHECKING:
    from agenticflow.capabilities.ssis.capability import SSISAnalyzer

logger = logging.getLogger(__name__)


class ExecuteProcessTaskHandler(TaskHandler):
    """Handler for Execute Process Task."""

    @property
    def task_patterns(self) -> list[str]:
        return ["executeprocess"]

    def handle(
        self,
        exe: ET.Element,
        analyzer: SSISAnalyzer,
        package_name: str,
        task_name: str,
    ) -> None:
        """Extract process execution details."""
        for obj_data in exe.iter():
            if obj_data.tag.endswith("ObjectData"):
                for child in obj_data:
                    executable = None
                    arguments = None
                    working_dir = None

                    for prop in child.iter():
                        prop_name = prop.get("Name") or prop.tag.split("}")[-1]
                        if "Executable" in prop_name:
                            executable = prop.text
                        elif "Arguments" in prop_name:
                            arguments = prop.text
                        elif "WorkingDirectory" in prop_name:
                            working_dir = prop.text

                    if executable:
                        # Update entity with process details
                        entity = analyzer.kg.graph.get_entity(task_name)
                        if entity:
                            entity.attributes["executable"] = executable
                            entity.attributes["arguments"] = arguments or ""
                            entity.attributes["working_directory"] = working_dir or ""

                        # Add relationship to external executable
                        exe_name = Path(executable).name if executable else "unknown"
                        analyzer.kg.graph.add_entity(
                            exe_name,
                            "ExternalProcess",
                            {"path": executable, "arguments": arguments},
                        )
                        analyzer.kg.graph.add_relationship(
                            task_name, "executes", exe_name
                        )


class ScriptTaskHandler(TaskHandler):
    """Handler for Script Task."""

    @property
    def task_patterns(self) -> list[str]:
        return ["scripttask"]

    def handle(
        self,
        exe: ET.Element,
        analyzer: SSISAnalyzer,
        package_name: str,
        task_name: str,
    ) -> None:
        """Extract script task details."""
        for obj_data in exe.iter():
            if obj_data.tag.endswith("ObjectData"):
                for child in obj_data:
                    script_lang = None
                    entry_point = None

                    for prop in child.iter():
                        prop_name = prop.get("Name") or prop.tag.split("}")[-1]
                        if "ScriptLanguage" in prop_name:
                            script_lang = prop.text
                        elif "EntryPoint" in prop_name:
                            entry_point = prop.text

                    entity = analyzer.kg.graph.get_entity(task_name)
                    if entity:
                        entity.attributes["script_language"] = script_lang or "VB"
                        entity.attributes["entry_point"] = entry_point or "Main"


class WebServiceTaskHandler(TaskHandler):
    """Handler for Web Service Task."""

    @property
    def task_patterns(self) -> list[str]:
        return ["webservice"]

    def handle(
        self,
        exe: ET.Element,
        analyzer: SSISAnalyzer,
        package_name: str,
        task_name: str,
    ) -> None:
        """Extract web service details."""
        for obj_data in exe.iter():
            if obj_data.tag.endswith("ObjectData"):
                for child in obj_data:
                    wsdl_url = None
                    method_name = None

                    for prop in child.iter():
                        prop_name = prop.get("Name") or prop.tag.split("}")[-1]
                        if "WsdlFile" in prop_name or "WSDL" in prop_name:
                            wsdl_url = prop.text
                        elif "MethodName" in prop_name or "WebMethod" in prop_name:
                            method_name = prop.text

                    if wsdl_url:
                        entity = analyzer.kg.graph.get_entity(task_name)
                        if entity:
                            entity.attributes["wsdl_url"] = wsdl_url
                            entity.attributes["method"] = method_name or ""

                        # Add web service as external entity
                        analyzer.kg.graph.add_entity(
                            wsdl_url,
                            "WebService",
                            {"url": wsdl_url, "method": method_name},
                        )
                        analyzer.kg.graph.add_relationship(
                            task_name, "calls_service", wsdl_url
                        )


class XMLTaskHandler(TaskHandler):
    """Handler for XML Task."""

    @property
    def task_patterns(self) -> list[str]:
        return ["xmltask"]

    def handle(
        self,
        exe: ET.Element,
        analyzer: SSISAnalyzer,
        package_name: str,
        task_name: str,
    ) -> None:
        """Extract XML task details."""
        for obj_data in exe.iter():
            if obj_data.tag.endswith("ObjectData"):
                for child in obj_data:
                    operation = None
                    source_type = None

                    for prop in child.iter():
                        prop_name = prop.get("Name") or prop.tag.split("}")[-1]
                        if "OperationType" in prop_name:
                            operation = prop.text
                        elif "SourceType" in prop_name:
                            source_type = prop.text

                    entity = analyzer.kg.graph.get_entity(task_name)
                    if entity:
                        entity.attributes["operation"] = operation or ""
                        entity.attributes["source_type"] = source_type or ""


# Default handlers to register
DEFAULT_HANDLERS: list[type[TaskHandler]] = [
    ExecuteProcessTaskHandler,
    ScriptTaskHandler,
    WebServiceTaskHandler,
    XMLTaskHandler,
]
