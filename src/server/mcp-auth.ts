import {
  CallToolRequestSchema,
  ReadResourceRequestSchema,
  GetPromptRequestSchema,
  CallToolRequest,
  ReadResourceRequest,
  GetPromptRequest,
  ListToolsRequestSchema,
  ListToolsResult,
  Tool,
  ListResourcesRequestSchema,
  Resource,
  ListPromptsRequestSchema,
  ListPromptsResult,
  Prompt,
  ListResourceTemplatesRequestSchema,
} from '../types.js';
import { RequestHandlerExtra } from '../shared/protocol.js';
import { z, ZodRawShape } from 'zod';
import {
  McpServer,
  ResourceMetadata,
  ReadResourceCallback,
  ReadResourceTemplateCallback,
  PromptCallback,
  ToolCallback,
  ResourceTemplate,
  promptArgumentsFromSchema,
} from './mcp.js';
import { zodToJsonSchema } from 'zod-to-json-schema';

export type AuthMethod = 'loginToken' | 'componentCall' | 'signedToken';

export interface SessionUser {
  did: string;
  role: string;
  provider: string;
  method?: AuthMethod;
  [key: string]: unknown;
}

export type AccessPolicy = {
  allow?: {
    dids?: string[];
    roles?: string[];
    providers?: string[];
    methods?: AuthMethod[];
  };
  deny?: {
    dids?: string[];
    roles?: string[];
    providers?: string[];
    methods?: AuthMethod[];
  };
};

interface RegisteredToolWithAuth {
  description?: string;
  inputSchema?: z.ZodObject<ZodRawShape>;
  callback: ToolCallback<ZodRawShape | undefined>;
  accessPolicy?: AccessPolicy;
}

interface RegisteredResourceWithAuth {
  name: string;
  metadata?: ResourceMetadata;
  readCallback: ReadResourceCallback;
  accessPolicy?: AccessPolicy;
}

interface RegisteredResourceTemplateWithAuth {
  resourceTemplate: ResourceTemplate;
  metadata?: ResourceMetadata;
  readCallback: ReadResourceTemplateCallback;
  accessPolicy?: AccessPolicy;
}

interface RegisteredPromptWithAuth {
  description?: string;
  argsSchema?: z.ZodObject<ZodRawShape>;
  callback: PromptCallback<ZodRawShape | undefined>;
  accessPolicy?: AccessPolicy;
}

export class McpServerWithAuth extends McpServer {
  protected override _registeredTools: { [name: string]: RegisteredToolWithAuth } = {};
  protected override _registeredResources: { [uri: string]: RegisteredResourceWithAuth } = {};
  protected override _registeredResourceTemplates: { [name: string]: RegisteredResourceTemplateWithAuth } = {};
  protected override _registeredPrompts: { [name: string]: RegisteredPromptWithAuth } = {};

  checkPermissions(user?: SessionUser, policy?: AccessPolicy): boolean {
    if (!policy) {
      return true;
    }

    if (!user) {
      return false;
    }

    // Check deny rules first
    if (policy.deny) {
      // Check denied DIDs
      if (policy.deny.dids?.includes(user.did)) {
        return false;
      }

      // Check denied roles
      if (policy.deny.roles?.includes(user.role)) {
        return false;
      }

      // Check denied providers
      if (policy.deny.providers?.includes(user.provider)) {
        return false;
      }

      // Check denied auth methods
      if (user.method && policy.deny.methods?.includes(user.method)) {
        return false;
      }
    }

    // Check allow rules
    if (policy.allow) {
      let isAllowed = false;

      // If no allow rules are specified, default to allowed
      if (!policy.allow.dids && !policy.allow.roles && !policy.allow.providers && !policy.allow.methods) {
        isAllowed = true;
      } else {
        // Check allowed DIDs
        if (policy.allow.dids?.includes(user.did)) {
          isAllowed = true;
        }

        // Check allowed roles
        if (policy.allow.roles?.includes(user.role)) {
          isAllowed = true;
        }

        // Check allowed providers
        if (policy.allow.providers?.includes(user.provider)) {
          isAllowed = true;
        }

        // Check allowed auth methods
        if (user.method && policy.allow.methods?.includes(user.method)) {
          isAllowed = true;
        }
      }

      return isAllowed;
    }

    // If no rules specified, default to allowed
    return true;
  }

  override tool(name: string, cb: ToolCallback, accessPolicy?: AccessPolicy): void;
  override tool(name: string, description: string, cb: ToolCallback, accessPolicy?: AccessPolicy): void;
  override tool<Args extends ZodRawShape>(
    name: string,
    paramsSchema: Args,
    cb: ToolCallback<Args>,
    accessPolicy?: AccessPolicy
  ): void;
  override tool<Args extends ZodRawShape>(
    name: string,
    description: string,
    paramsSchema: Args,
    cb: ToolCallback<Args>,
    accessPolicy?: AccessPolicy
  ): void;
  override tool(name: string, ...rest: unknown[]): void {
    let description: string | undefined;
    let paramsSchema: ZodRawShape | undefined;
    let accessPolicy: AccessPolicy | undefined;
    let cb: ToolCallback<ZodRawShape | undefined>;

    // Parse arguments based on their types
    if (typeof rest[0] === 'function') {
      // Case: tool(name, cb, accessPolicy?)
      cb = rest[0] as ToolCallback<ZodRawShape | undefined>;
      accessPolicy = rest[1] as AccessPolicy | undefined;
    } else if (typeof rest[0] === 'string') {
      // Cases with description
      description = rest[0];
      if (typeof rest[1] === 'function') {
        // Case: tool(name, description, cb, accessPolicy?)
        cb = rest[1] as ToolCallback<ZodRawShape | undefined>;
        accessPolicy = rest[2] as AccessPolicy | undefined;
      } else {
        // Case: tool(name, description, paramsSchema, cb, accessPolicy?)
        paramsSchema = rest[1] as ZodRawShape;
        cb = rest[2] as ToolCallback<ZodRawShape>;
        accessPolicy = rest[3] as AccessPolicy | undefined;
      }
    } else {
      // Case: tool(name, paramsSchema, cb, accessPolicy?)
      paramsSchema = rest[0] as ZodRawShape;
      cb = rest[1] as ToolCallback<ZodRawShape>;
      accessPolicy = rest[2] as AccessPolicy | undefined;
    }

    // Register with base class
    const args: unknown[] = [name];
    if (description) args.push(description);
    if (paramsSchema) args.push(paramsSchema);
    args.push(cb);

    // Set up request handlers if not already initialized
    if (!this._toolHandlersInitialized) {
      this.server.assertCanSetRequestHandler(CallToolRequestSchema.shape.method.value);
      this.server.assertCanSetRequestHandler(ListToolsRequestSchema.shape.method.value);
      this.server.registerCapabilities({ tools: {} });

      // Add ListToolsRequestSchema handler
      this.server.setRequestHandler(ListToolsRequestSchema, (request, extra): ListToolsResult => {
        const user = extra.user as SessionUser | undefined;

        console.info('list tools', { request, extra });

        // Filter tools based on permissions
        const accessibleTools = Object.entries(this._registeredTools)
          .filter(([_, tool]) => this.checkPermissions(user, tool.accessPolicy))
          .map(
            ([name, tool]): Tool => ({
              name,
              description: tool.description,
              inputSchema: tool.inputSchema
                ? (zodToJsonSchema(tool.inputSchema, {
                    strictUnions: true,
                  }) as Tool['inputSchema'])
                : { type: 'object' },
            })
          );

        return { tools: accessibleTools };
      });

      this.server.setRequestHandler(
        CallToolRequestSchema,
        async (request: CallToolRequest, extra: RequestHandlerExtra) => {
          console.info('call tool', { request, extra });

          const tool = this._registeredTools[request.params.name];
          if (!tool) {
            throw new Error(`Tool ${request.params.name} not found`);
          }

          if (!this.checkPermissions(extra.user as SessionUser, tool.accessPolicy)) {
            throw new Error(`Access denied for tool: ${request.params.name}`);
          }

          if (tool.inputSchema) {
            const parseResult = await tool.inputSchema.safeParseAsync(request.params.arguments);
            if (!parseResult.success) {
              throw new Error(`Invalid arguments for tool ${request.params.name}: ${parseResult.error.message}`);
            }

            const args = parseResult.data;
            const cb = tool.callback as ToolCallback<ZodRawShape>;
            return await Promise.resolve(cb(args, extra));
          } else {
            const cb = tool.callback as ToolCallback<undefined>;
            return await Promise.resolve(cb(extra));
          }
        }
      );
      this._toolHandlersInitialized = true;
    }

    McpServer.prototype.tool.apply(this, args as Parameters<typeof McpServer.prototype.tool>);
    this._registeredTools[name].accessPolicy = accessPolicy;
  }

  override resource(name: string, uri: string, readCallback: ReadResourceCallback, accessPolicy?: AccessPolicy): void;
  override resource(
    name: string,
    uri: string,
    metadata: ResourceMetadata,
    readCallback: ReadResourceCallback,
    accessPolicy?: AccessPolicy
  ): void;
  override resource(
    name: string,
    template: ResourceTemplate,
    readCallback: ReadResourceTemplateCallback,
    accessPolicy?: AccessPolicy
  ): void;
  override resource(
    name: string,
    template: ResourceTemplate,
    metadata: ResourceMetadata,
    readCallback: ReadResourceTemplateCallback,
    accessPolicy?: AccessPolicy
  ): void;
  override resource(name: string, uriOrTemplate: string | ResourceTemplate, ...rest: unknown[]): void {
    let metadata: ResourceMetadata | undefined;
    let accessPolicy: AccessPolicy | undefined;
    let readCallback: ReadResourceCallback | ReadResourceTemplateCallback;

    // Parse arguments based on their types
    if (typeof rest[0] === 'function') {
      // Case: resource(name, uri|template, readCallback, accessPolicy?)
      readCallback = rest[0] as ReadResourceCallback | ReadResourceTemplateCallback;
      accessPolicy = rest[1] as AccessPolicy | undefined;
    } else {
      // Case: resource(name, uri|template, metadata, readCallback, accessPolicy?)
      metadata = rest[0] as ResourceMetadata;
      readCallback = rest[1] as ReadResourceCallback | ReadResourceTemplateCallback;
      accessPolicy = rest[2] as AccessPolicy | undefined;
    }

    // Register with base class
    const args: unknown[] = [name, uriOrTemplate];
    if (metadata) args.push(metadata);
    args.push(readCallback);

    // Set up request handlers if not already initialized
    if (!this._resourceHandlersInitialized) {
      this.server.assertCanSetRequestHandler(ReadResourceRequestSchema.shape.method.value);
      this.server.assertCanSetRequestHandler(ListResourcesRequestSchema.shape.method.value);
      this.server.assertCanSetRequestHandler(ListResourceTemplatesRequestSchema.shape.method.value);
      this.server.registerCapabilities({ resources: {} });

      // Add ListResourcesRequestSchema handler
      this.server.setRequestHandler(ListResourcesRequestSchema, async (request, extra) => {
        const user = extra.user as SessionUser | undefined;

        // Filter static resources based on permissions
        const staticResources = Object.entries(this._registeredResources)
          .filter(([_, resource]) => this.checkPermissions(user, resource.accessPolicy))
          .map(
            ([uri, resource]): Resource => ({
              uri,
              name: resource.name,
              ...resource.metadata,
            })
          );

        // Filter and get template resources based on permissions
        const templateResources: Resource[] = [];
        for (const template of Object.values(this._registeredResourceTemplates)) {
          if (!this.checkPermissions(user, template.accessPolicy)) {
            continue;
          }

          if (!template.resourceTemplate.listCallback) {
            continue;
          }

          const result = await template.resourceTemplate.listCallback(extra);
          for (const resource of result.resources) {
            templateResources.push({
              ...resource,
              ...template.metadata,
            });
          }
        }

        return { resources: [...staticResources, ...templateResources] };
      });

      // Add ListResourceTemplatesRequestSchema handler
      this.server.setRequestHandler(ListResourceTemplatesRequestSchema, async (request, extra) => {
        const user = extra.user as SessionUser | undefined;

        const resourceTemplates = Object.entries(this._registeredResourceTemplates)
          .filter(([_, template]) => this.checkPermissions(user, template.accessPolicy))
          .map(([name, template]) => ({
            name,
            uriTemplate: template.resourceTemplate.uriTemplate.toString(),
            ...template.metadata,
          }));

        return { resourceTemplates };
      });

      // Add ReadResourceRequestSchema handler
      this.server.setRequestHandler(
        ReadResourceRequestSchema,
        async (request: ReadResourceRequest, extra: RequestHandlerExtra) => {
          const uri = new URL(request.params.uri);

          // First check for exact resource match
          const resource = this._registeredResources[uri.toString()];
          if (resource) {
            if (!this.checkPermissions(extra.user as SessionUser, resource.accessPolicy)) {
              throw new Error('Access denied for resource');
            }
            return resource.readCallback(uri, extra);
          }

          // Then check templates
          for (const template of Object.values(this._registeredResourceTemplates)) {
            const variables = template.resourceTemplate.uriTemplate.match(uri.toString());
            if (variables) {
              if (!this.checkPermissions(extra.user as SessionUser, template.accessPolicy)) {
                throw new Error('Access denied for resource template');
              }
              return template.readCallback(uri, variables, extra);
            }
          }

          throw new Error(`Resource ${uri} not found`);
        }
      );
      this._resourceHandlersInitialized = true;
    }

    McpServer.prototype.resource.apply(this, args as Parameters<typeof McpServer.prototype.resource>);
    if (typeof uriOrTemplate === 'string') {
      this._registeredResources[uriOrTemplate].accessPolicy = accessPolicy;
    } else {
      this._registeredResourceTemplates[name].accessPolicy = accessPolicy;
    }
  }

  override prompt(name: string, cb: PromptCallback, accessPolicy?: AccessPolicy): void;
  override prompt(name: string, description: string, cb: PromptCallback, accessPolicy?: AccessPolicy): void;
  override prompt<Args extends ZodRawShape>(
    name: string,
    argsSchema: Args,
    cb: PromptCallback<Args>,
    accessPolicy?: AccessPolicy
  ): void;
  override prompt<Args extends ZodRawShape>(
    name: string,
    description: string,
    argsSchema: Args,
    cb: PromptCallback<Args>,
    accessPolicy?: AccessPolicy
  ): void;
  override prompt(name: string, ...rest: unknown[]): void {
    let description: string | undefined;
    let argsSchema: ZodRawShape | undefined;
    let accessPolicy: AccessPolicy | undefined;
    let cb: PromptCallback<ZodRawShape | undefined>;

    // Parse arguments based on their types
    if (typeof rest[0] === 'function') {
      // Case: prompt(name, cb, accessPolicy?)
      cb = rest[0] as PromptCallback<ZodRawShape | undefined>;
      accessPolicy = rest[1] as AccessPolicy | undefined;
    } else if (typeof rest[0] === 'string') {
      // Cases with description
      description = rest[0];
      if (typeof rest[1] === 'function') {
        // Case: prompt(name, description, cb, accessPolicy?)
        cb = rest[1] as PromptCallback<ZodRawShape | undefined>;
        accessPolicy = rest[2] as AccessPolicy | undefined;
      } else {
        // Case: prompt(name, description, argsSchema, cb, accessPolicy?)
        argsSchema = rest[1] as ZodRawShape;
        cb = rest[2] as PromptCallback<ZodRawShape>;
        accessPolicy = rest[3] as AccessPolicy | undefined;
      }
    } else {
      // Case: prompt(name, argsSchema, cb, accessPolicy?)
      argsSchema = rest[0] as ZodRawShape;
      cb = rest[1] as PromptCallback<ZodRawShape>;
      accessPolicy = rest[2] as AccessPolicy | undefined;
    }

    // Register with base class
    const args: unknown[] = [name];
    if (description) args.push(description);
    if (argsSchema) args.push(argsSchema);
    args.push(cb);

    // Set up request handlers if not already initialized
    if (!this._promptHandlersInitialized) {
      this.server.assertCanSetRequestHandler(GetPromptRequestSchema.shape.method.value);
      this.server.assertCanSetRequestHandler(ListPromptsRequestSchema.shape.method.value);
      this.server.registerCapabilities({ prompts: {} });

      // Add ListPromptsRequestSchema handler
      this.server.setRequestHandler(ListPromptsRequestSchema, (request, extra): ListPromptsResult => {
        const user = extra.user as SessionUser | undefined;

        // Filter prompts based on permissions
        const accessiblePrompts = Object.entries(this._registeredPrompts)
          .filter(([_, prompt]) => this.checkPermissions(user, prompt.accessPolicy))
          .map(
            ([name, prompt]): Prompt => ({
              name,
              description: prompt.description,
              arguments: prompt.argsSchema ? promptArgumentsFromSchema(prompt.argsSchema) : undefined,
            })
          );

        return { prompts: accessiblePrompts };
      });

      // Add GetPromptRequestSchema handler
      this.server.setRequestHandler(
        GetPromptRequestSchema,
        async (request: GetPromptRequest, extra: RequestHandlerExtra) => {
          const prompt = this._registeredPrompts[request.params.name];
          if (!prompt) {
            throw new Error(`Prompt ${request.params.name} not found`);
          }

          if (!this.checkPermissions(extra.user as SessionUser, prompt.accessPolicy)) {
            throw new Error('Access denied for prompt');
          }

          if (prompt.argsSchema) {
            const parseResult = await prompt.argsSchema.safeParseAsync(request.params.arguments);
            if (!parseResult.success) {
              throw new Error(`Invalid arguments for prompt ${request.params.name}: ${parseResult.error.message}`);
            }

            const args = parseResult.data;
            const cb = prompt.callback as PromptCallback<ZodRawShape>;
            return await Promise.resolve(cb(args, extra));
          } else {
            const cb = prompt.callback as PromptCallback<undefined>;
            return await Promise.resolve(cb(extra));
          }
        }
      );
      this._promptHandlersInitialized = true;
    }

    McpServer.prototype.prompt.apply(this, args as Parameters<typeof McpServer.prototype.prompt>);
    this._registeredPrompts[name].accessPolicy = accessPolicy;
  }
}
